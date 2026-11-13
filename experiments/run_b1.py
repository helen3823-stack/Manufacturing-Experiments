
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Dataset
from src.evaluate import report
from src.metrics import f1

CATEGORIES = [
    "elec_01", "mech_05", "agri_02",
    "med_08", "groc_01"
]

BANK_SIZE = 10000
SEED = 42
OUT = ROOT / "artifacts" / "experiments" / "B1"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class B1Model:
    def __init__(self, device):
        self.device = device
        self.net = resnet18()

        weights = ROOT / "artifacts" / "resnet18.pt"
        if not weights.exists():
            from torchvision.models import ResNet18_Weights
            pretrained = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pretrained.state_dict(), weights)
            del pretrained

        self.net.load_state_dict(
            torch.load(
                weights,
                map_location="cpu",
                weights_only=True
            )
        )
        self.net = self.net.to(device).eval()

        self.mean = torch.tensor(
            MEAN, device=device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            STD, device=device
        ).view(1, 3, 1, 1)

        self.bank = None
        self.bank_sq = None

    @torch.inference_mode()
    def features(self, images):
        x = torch.from_numpy(
            np.ascontiguousarray(images)
        ).to(self.device)

        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std

        net = self.net
        x = net.maxpool(
            net.relu(net.bn1(net.conv1(x)))
        )
        x = net.layer1(x)

        f2 = net.layer2(x)
        f3 = net.layer3(f2)

        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        f = torch.cat([f2, f3], dim=1)

        # Neighborhood 3: 주변 3x3 특징 평균
        f = F.avg_pool2d(
            F.pad(f, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1
        )

        return f.permute(0, 2, 3, 1).contiguous()

    def fit(self, cat, limit=None):
        rng = np.random.default_rng(SEED)

        # 무작위 키를 이용한 균등 패치 샘플링
        # 전체 100만 개 이상의 패치를 저장하지 않음
        selected = np.empty((0, 384), dtype=np.float32)
        selected_keys = np.empty(0, dtype=np.float64)

        for batch_idx, (_, images) in enumerate(
            cat.batches("train", batch=8, limit=limit)
        ):
            f = self.features(images)
            flat = f.reshape(-1, 384).cpu().numpy()
            keys = rng.random(len(flat))

            combined = np.concatenate(
                [selected, flat], axis=0
            )
            combined_keys = np.concatenate(
                [selected_keys, keys]
            )

            keep_n = min(BANK_SIZE, len(combined_keys))

            if keep_n == len(combined_keys):
                keep = np.arange(keep_n)
            else:
                keep = np.argpartition(
                    combined_keys, keep_n - 1
                )[:keep_n]

            selected = combined[keep].copy()
            selected_keys = combined_keys[keep].copy()

            if (batch_idx + 1) % 25 == 0:
                print(
                    f"  Train 배치 {batch_idx + 1} 완료",
                    flush=True
                )

        if len(selected) == 0:
            raise ValueError("Train 데이터가 비어 있습니다.")

        self.bank = torch.from_numpy(
            np.ascontiguousarray(selected)
        ).to(self.device)

        self.bank_sq = (
            self.bank.square().sum(dim=1)
        )

        print(
            f"  Memory Bank: {tuple(self.bank.shape)}",
            flush=True
        )

    @torch.inference_mode()
    def predict_batch(self, images):
        f = self.features(images)
        n, h, w, d = f.shape
        q = f.reshape(-1, d)

        distances = []

        # RTX 2070 8GB VRAM 보호: 작은 청크 사용
        for start in range(0, len(q), 128):
            x = q[start:start + 128]

            dist_sq = (
                x.square().sum(dim=1, keepdim=True)
                + self.bank_sq.unsqueeze(0)
                - 2 * x @ self.bank.T
            )

            nearest = torch.sqrt(
                torch.clamp(
                    dist_sq.min(dim=1).values,
                    min=0
                )
            )
            distances.append(nearest)

        grid = torch.cat(distances).reshape(
            n, 1, h, w
        )

        maps = F.interpolate(
            grid,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        maps = maps[:, 0].cpu().numpy()

        maps = gaussian_filter(
            maps,
            sigma=(0, 4, 4)
        ).astype(np.float32)

        flat = maps.reshape(n, -1)
        k = max(1, int(flat.shape[1] * 0.01))

        scores = np.partition(
            flat, -k, axis=1
        )[:, -k:].mean(axis=1)

        return maps, scores

    def save(self, folder, threshold):
        folder.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            folder / "model.npz",
            bank=self.bank.cpu().numpy(),
            threshold=float(threshold),
        )


def optimal_threshold(labels, scores):
    best_t = float(scores.min())
    best_f = -1.0

    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        current = f1(labels, pred)["f1"]

        if current > best_f:
            best_f = current
            best_t = float(t)

    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device, flush=True)

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA가 비활성화되어 있습니다. GPU 환경을 확인하세요."
        )

    ds = Dataset()
    results = []

    for name in args.categories:
        print(f"\n[{name}] B1 시작", flush=True)
        cat = ds.category(name)

        model = B1Model(device)

        start = time.time()
        model.fit(
            cat,
            limit=32 if args.smoke else None
        )
        fit_sec = time.time() - start

        if args.smoke:
            images = cat.images(cat.val.head(2))
            maps, scores = model.predict_batch(images)

            print("Maps:", maps.shape)
            print("Scores:", scores.shape)
            print("Smoke Test 성공!")
            del model
            continue

        val = cat.val.copy()
        all_maps = []
        all_scores = []

        start = time.time()

        for batch_idx, (_, images) in enumerate(
            cat.batches(val, batch=4)
        ):
            maps, scores = model.predict_batch(images)
            all_maps.append(maps)
            all_scores.append(scores)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Val 배치 {batch_idx + 1} 완료",
                    flush=True
                )

        infer_sec = time.time() - start

        maps = np.concatenate(all_maps)
        scores = np.concatenate(all_scores)

        thr = optimal_threshold(
            val.is_anomaly.to_numpy(), scores
        )

        val["score"] = scores
        val["pred"] = (scores >= thr).astype(int)

        masks = cat.masks(val)
        metrics = report(val, masks, maps)

        official = np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"]
        ])

        model.save(OUT / name, thr)

        results.append({
            "category": name,
            "threshold": thr,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec,
            "bank_size": BANK_SIZE,
        })

        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(
            OUT / "results.csv",
            index=False
        )

        print(
            f"  AUROC={metrics['image_auroc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"AUPRO={metrics['pixel_aupro']:.4f} "
            f"Score={official:.4f}",
            flush=True
        )

        del model, maps, masks
        torch.cuda.empty_cache()

    if results:
        df = pd.DataFrame(results)

        final = df[
            ["image_auroc", "f1", "pixel_aupro"]
        ].mean().mean()

        print("\n===== B1 결과 =====")
        print(df[
            ["category", "image_auroc",
             "f1", "pixel_aupro", "official_score",
             "fit_sec", "inference_sec"]
        ].to_string(index=False))

        print(f"\n대표 5개 평균: {final:.4f}")


if __name__ == "__main__":
    main()
