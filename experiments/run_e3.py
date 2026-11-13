
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import run_e1 as e1
from run_b1 import ROOT, CATEGORIES, SEED
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E3"
BASELINE = ROOT / "artifacts" / "experiments" / "E2" / "bank_5000" / "results.csv"

BANK_SIZE = 5000
CANDIDATES = 30000
PROJ_DIM = 64

LAYER_OPTIONS = {
    "L1L2": ("layer1", "layer2"),
    "L1L2L3": ("layer1", "layer2", "layer3"),
}


class E3Model(e1.E1Model):
    def __init__(self, device, layer_option):
        super().__init__(device)
        self.layer_option = layer_option
        self.captures = {}
        self.handles = []

        # B1 내부에 들어 있는 torchvision ResNet18 탐색
        backbone = None
        for obj in vars(self).values():
            if isinstance(obj, nn.Module):
                for module in obj.modules():
                    if all(
                        hasattr(module, x)
                        for x in ("layer1", "layer2", "layer3")
                    ):
                        backbone = module
                        break
            if backbone is not None:
                break

        if backbone is None:
            raise RuntimeError(
                "B1 내부 ResNet18을 찾지 못했습니다. "
                "run_b1.py의 backbone 구조 확인이 필요합니다."
            )

        self.feature_dim = sum(
            {"layer1": 64, "layer2": 128, "layer3": 256}[x]
            for x in LAYER_OPTIONS[layer_option]
        )

        def make_hook(name):
            def hook(module, inputs, output):
                self.captures[name] = output
            return hook

        for name in LAYER_OPTIONS[layer_option]:
            self.handles.append(
                getattr(backbone, name).register_forward_hook(
                    make_hook(name)
                )
            )

    @torch.inference_mode()
    def features(self, images):
        self.captures.clear()

        # 기존 B1의 이미지 전처리 및 ResNet forward 재사용
        _ = super().features(images)

        tensors = []
        for name in LAYER_OPTIONS[self.layer_option]:
            if name not in self.captures:
                raise RuntimeError(f"{name} 출력 캡처 실패")

            x = self.captures[name]
            x = F.interpolate(
                x,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(x)

        combined = torch.cat(tensors, dim=1)

        # Neighborhood = 3
        combined = F.avg_pool2d(
            F.pad(combined, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )

        return combined.permute(0, 2, 3, 1).contiguous()

    @torch.inference_mode()
    def fit(self, cat, limit=None):
        rng = np.random.default_rng(SEED)
        selected = np.empty(
            (0, self.feature_dim), dtype=np.float32
        )
        selected_keys = np.empty(0, dtype=np.float64)

        # E1과 동일한 random-key 방식으로 후보 30k 선정
        for _, images in cat.batches(
            "train", batch=8, limit=limit
        ):
            flat = self.features(images)
            flat = flat.reshape(
                -1, self.feature_dim
            ).cpu().numpy()

            keys = rng.random(len(flat))
            combined = np.concatenate(
                [selected, flat], axis=0
            )
            combined_keys = np.concatenate(
                [selected_keys, keys]
            )

            n = min(CANDIDATES, len(combined_keys))
            if n == len(combined_keys):
                keep = np.arange(n)
            else:
                keep = np.argpartition(
                    combined_keys, n - 1
                )[:n]

            selected = combined[keep].copy()
            selected_keys = combined_keys[keep].copy()

        print(
            f"  Candidates: {len(selected)}"
            f" | Feature dim: {self.feature_dim}",
            flush=True
        )

        # Random Projection
        generator = torch.Generator().manual_seed(SEED)
        projection = torch.randn(
            self.feature_dim,
            PROJ_DIM,
            generator=generator
        ) / (PROJ_DIM ** 0.5)

        projection = projection.to(self.device)
        candidates = torch.from_numpy(
            np.ascontiguousarray(selected)
        ).to(self.device)

        projected = candidates @ projection
        norms = projected.square().sum(dim=1)

        # Greedy Farthest-first
        k = min(BANK_SIZE, len(selected))
        min_dist = torch.full(
            (len(selected),),
            float("inf"),
            device=self.device
        )
        chosen = torch.zeros(
            len(selected),
            dtype=torch.bool,
            device=self.device
        )
        indices = torch.empty(
            k,
            dtype=torch.long,
            device=self.device
        )
        idx = torch.tensor(
            int(rng.integers(len(selected))),
            device=self.device
        )

        for i in range(k):
            indices[i] = idx
            chosen[idx] = True

            center = projected[idx]
            dist = (
                norms + norms[idx]
                - 2 * (projected @ center)
            ).clamp_min_(0)

            min_dist = torch.minimum(min_dist, dist)
            min_dist.masked_fill_(chosen, -1)

            if i + 1 < k:
                idx = min_dist.argmax()

            if (i + 1) % 1000 == 0:
                print(f"  Greedy {i+1}/{k}", flush=True)

        bank = np.ascontiguousarray(
            selected[indices.cpu().numpy()]
        )

        self.bank = torch.from_numpy(bank).to(
            self.device
        )
        self.bank_sq = self.bank.square().sum(dim=1)

        print(
            f"  Final bank: {tuple(self.bank.shape)}",
            flush=True
        )
        return self

    @torch.inference_mode()
    def predict_batch(self, images):
        features = self.features(images)
        b, h, w, d = features.shape

        queries = features.reshape(-1, d)
        distances = []

        # GPU 메모리 절약을 위한 chunk 계산
        for chunk in queries.split(128):
            q_sq = chunk.square().sum(
                dim=1, keepdim=True
            )
            dist_sq = (
                q_sq + self.bank_sq[None, :]
                - 2 * (chunk @ self.bank.T)
            ).clamp_min_(0)

            distances.append(
                dist_sq.min(dim=1).values.sqrt()
            )

        anomaly = torch.cat(distances)
        anomaly = anomaly.reshape(b, 1, h, w)

        anomaly = F.interpolate(
            anomaly,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        maps = anomaly[:, 0].cpu().numpy()
        maps = gaussian_filter(
            maps, sigma=(0, 4, 4)
        )

        # 이미지 점수 = 상위 1% pixel 평균
        pixels = maps.reshape(b, -1)
        k = max(1, int(np.ceil(pixels.shape[1] * 0.01)))
        scores = np.partition(
            pixels, -k, axis=1
        )[:, -k:].mean(axis=1)

        return maps, scores

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=list(LAYER_OPTIONS),
        default=list(LAYER_OPTIONS)
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다.")

    torch.manual_seed(SEED)
    ds = Dataset()
    device = torch.device("cuda")

    for option in args.layers:
        folder = OUT / option
        folder.mkdir(parents=True, exist_ok=True)
        results = []

        for name in args.categories:
            print(
                f"\n[{name}] E3 {option} 시작",
                flush=True
            )

            cat = ds.category(name)
            model = E3Model(device, option)

            start = time.time()
            model.fit(
                cat,
                limit=32 if args.smoke else None
            )
            torch.cuda.synchronize()
            fit_sec = time.time() - start

            if args.smoke:
                maps, scores = model.predict_batch(
                    cat.images(cat.val.head(2))
                )
                print(
                    "Smoke OK:",
                    maps.shape,
                    scores.shape
                )
                model.close()
                del model
                torch.cuda.empty_cache()
                continue

            val = cat.val.copy()
            maps_list, scores_list = [], []

            start = time.time()
            for _, images in cat.batches(val, batch=4):
                maps, scores = model.predict_batch(images)
                maps_list.append(maps)
                scores_list.append(scores)

            torch.cuda.synchronize()
            infer_sec = time.time() - start

            maps = np.concatenate(maps_list)
            scores = np.concatenate(scores_list)

            threshold = e1.optimal_threshold(
                val.is_anomaly.to_numpy(),
                scores
            )

            val["score"] = scores
            val["pred"] = (
                scores >= threshold
            ).astype(int)

            metrics = report(
                val, cat.masks(val), maps
            )

            official = np.mean([
                metrics["image_auroc"],
                metrics["f1"],
                metrics["pixel_aupro"]
            ])

            model.save(folder / name, threshold)

            results.append({
                "category": name,
                "layers": option,
                "feature_dim": model.feature_dim,
                "bank_size": BANK_SIZE,
                "threshold": threshold,
                **metrics,
                "official_score": official,
                "fit_sec": fit_sec,
                "inference_sec": infer_sec,
            })

            pd.DataFrame(results).to_csv(
                folder / "results.csv",
                index=False
            )

            print(
                f"  AUROC={metrics['image_auroc']:.4f}"
                f" F1={metrics['f1']:.4f}"
                f" AUPRO={metrics['pixel_aupro']:.4f}"
                f" Score={official:.4f}",
                flush=True
            )

            model.close()
            del model, maps
            torch.cuda.empty_cache()

    if args.smoke:
        return

    frames = []

    if BASELINE.exists():
        base = pd.read_csv(BASELINE)
        base = base[
            base["category"].isin(args.categories)
        ].copy()
        base["layers"] = "L2L3"
        frames.append(base)

    for option in args.layers:
        path = OUT / option / "results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    if frames:
        comparison = pd.concat(
            frames, ignore_index=True
        )
        OUT.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            OUT / "comparison.csv",
            index=False
        )

        print("\n===== E3 레이어별 평균 =====")
        print(
            comparison.groupby("layers")[
                ["image_auroc", "f1",
                 "pixel_aupro", "official_score"]
            ].mean().round(4).to_string()
        )

        print("\n===== 카테고리별 점수 =====")
        print(
            comparison.pivot_table(
                index="category",
                columns="layers",
                values="official_score"
            ).round(4).to_string()
        )


if __name__ == "__main__":
    main()
