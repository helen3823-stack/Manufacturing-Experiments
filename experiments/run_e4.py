
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import run_e3 as e3
from run_b1 import ROOT, CATEGORIES, SEED
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E4"
BASELINE = (
    ROOT / "artifacts" / "experiments"
    / "E3" / "L1L2L3" / "results.csv"
)


class E4Model(e3.E3Model):
    def __init__(self, device, neighborhood):
        super().__init__(device, "L1L2L3")
        self.neighborhood = neighborhood

    @torch.inference_mode()
    def features(self, images):
        self.captures.clear()

        # E3와 동일한 backbone 및 전처리
        # Hook을 통해 L1, L2, L3 출력 확보
        _ = e3.e1.E1Model.features(self, images)

        tensors = []
        for name in ("layer1", "layer2", "layer3"):
            x = self.captures[name]
            x = F.interpolate(
                x,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(x)

        x = torch.cat(tensors, dim=1)

        # E4에서 변경하는 유일한 요소
        n = self.neighborhood

        if n > 1:
            padding = n // 2
            x = F.avg_pool2d(
                F.pad(
                    x,
                    (padding,) * 4,
                    mode="reflect",
                ),
                kernel_size=n,
                stride=1,
            )

        return x.permute(0, 2, 3, 1).contiguous()


def run_one(neighborhood, categories, smoke=False):
    folder = OUT / f"neighborhood_{neighborhood}"
    folder.mkdir(parents=True, exist_ok=True)

    ds = Dataset()
    device = torch.device("cuda")
    results = []

    for name in categories:
        print(
            f"\n[{name}] E4 Neighborhood {neighborhood} 시작",
            flush=True,
        )

        cat = ds.category(name)
        model = E4Model(device, neighborhood)

        t0 = time.time()
        model.fit(cat, limit=32 if smoke else None)
        torch.cuda.synchronize()
        fit_sec = time.time() - t0

        if smoke:
            maps, scores = model.predict_batch(
                cat.images(cat.val.head(2))
            )
            print("Smoke OK:", maps.shape, scores.shape)
            model.close()
            del model
            torch.cuda.empty_cache()
            continue

        val = cat.val.copy()
        maps_list = []
        scores_list = []

        t1 = time.time()
        for _, images in cat.batches(val, batch=4):
            maps, scores = model.predict_batch(images)
            maps_list.append(maps)
            scores_list.append(scores)

        torch.cuda.synchronize()
        infer_sec = time.time() - t1

        maps = np.concatenate(maps_list)
        scores = np.concatenate(scores_list)

        threshold = e3.e1.optimal_threshold(
            val.is_anomaly.to_numpy(),
            scores,
        )

        val["score"] = scores
        val["pred"] = (scores >= threshold).astype(int)

        metrics = report(val, cat.masks(val), maps)

        official = np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"],
        ])

        model.save(folder / name, threshold)

        results.append({
            "category": name,
            "neighborhood": neighborhood,
            "feature_dim": model.feature_dim,
            "bank_size": 5000,
            "threshold": threshold,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec,
        })

        pd.DataFrame(results).to_csv(
            folder / "results.csv", index=False
        )

        print(
            f"  AUROC={metrics['image_auroc']:.4f}"
            f" F1={metrics['f1']:.4f}"
            f" AUPRO={metrics['pixel_aupro']:.4f}"
            f" Score={official:.4f}",
            flush=True,
        )

        model.close()
        del model, maps
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", nargs="+", default=CATEGORIES
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1, 5]
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    if any(n not in (1, 5) for n in args.sizes):
        raise ValueError("Neighborhood 1 또는 5를 선택하세요.")

    torch.manual_seed(SEED)

    for n in args.sizes:
        run_one(n, args.categories, args.smoke)

    if args.smoke:
        return

    frames = []

    if BASELINE.exists():
        baseline = pd.read_csv(BASELINE)
        baseline = baseline[
            baseline["category"].isin(args.categories)
        ].copy()
        baseline["neighborhood"] = 3
        frames.append(baseline)
    else:
        print("주의: E3 기준 결과 파일이 없습니다.")

    for n in (1, 5):
        path = (
            OUT / f"neighborhood_{n}" / "results.csv"
        )
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame)

    if frames:
        comparison = pd.concat(
            frames, ignore_index=True
        )
        comparison.to_csv(
            OUT / "comparison.csv", index=False
        )

        print("\n===== E4 Neighborhood별 평균 =====")
        print(
            comparison.groupby("neighborhood")[
                [
                    "image_auroc", "f1",
                    "pixel_aupro", "official_score"
                ]
            ].mean().round(4).to_string()
        )

        print("\n===== 카테고리별 종합 점수 =====")
        print(
            comparison.pivot_table(
                index="category",
                columns="neighborhood",
                values="official_score",
            ).round(4).to_string()
        )


if __name__ == "__main__":
    main()
