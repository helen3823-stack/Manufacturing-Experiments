
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import run_e5 as e5
from run_b1 import ROOT, CATEGORIES, SEED
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E6"
BASELINE = (
    ROOT / "artifacts" / "experiments"
    / "E5" / "l2" / "results.csv"
)


class E6Model(e5.E5Model):
    def __init__(self, device):
        super().__init__(device, mode="l2")

    @torch.inference_mode()
    def predict_batch(self, images):
        # E5와 동일한 L2 정규화 특징 추출
        features = self.features(images)
        b, h, w, d = features.shape

        queries = features.reshape(-1, d)
        distances = []

        # E5와 동일한 Euclidean 1-NN 계산
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

        # E6에서 변경하는 유일한 요소:
        # bilinear -> nearest
        anomaly = F.interpolate(
            anomaly,
            size=(256, 256),
            mode="nearest",
        )

        maps = anomaly[:, 0].cpu().numpy()

        # 기존 E5와 동일한 Gaussian smoothing
        maps = gaussian_filter(
            maps, sigma=(0, 4, 4)
        )

        # 기존 E5와 동일한 Top 1% 이미지 점수
        pixels = maps.reshape(b, -1)
        k = max(
            1, int(np.ceil(pixels.shape[1] * 0.01))
        )

        scores = np.partition(
            pixels, -k, axis=1
        )[:, -k:].mean(axis=1)

        return maps, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    torch.manual_seed(SEED)

    ds = Dataset()
    device = torch.device("cuda")

    folder = OUT / "nearest"
    folder.mkdir(parents=True, exist_ok=True)

    results = []

    for name in args.categories:
        print(
            f"\n[{name}] E6 Nearest 시작",
            flush=True,
        )

        cat = ds.category(name)
        model = E6Model(device)

        start = time.time()
        model.fit(
            cat, limit=32 if args.smoke else None
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
                scores.shape,
                flush=True,
            )
            model.close()
            del model
            torch.cuda.empty_cache()
            continue

        val = cat.val.copy()
        maps_list = []
        scores_list = []

        start = time.time()

        for _, images in cat.batches(val, batch=4):
            maps, scores = model.predict_batch(images)
            maps_list.append(maps)
            scores_list.append(scores)

        torch.cuda.synchronize()
        infer_sec = time.time() - start

        maps = np.concatenate(maps_list)
        scores = np.concatenate(scores_list)

        # 기존 E5와 동일한 Validation F1 임계값 선택
        threshold = e5.e4.e3.e1.optimal_threshold(
            val.is_anomaly.to_numpy(),
            scores,
        )

        val["score"] = scores
        val["pred"] = (
            scores >= threshold
        ).astype(int)

        metrics = report(
            val, cat.masks(val), maps
        )

        official = float(np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"],
        ]))

        model.save(folder / name, threshold)

        results.append({
            "category": name,
            "upsampling": "nearest",
            "bank_size": 5000,
            "neighborhood": 5,
            "normalization": "l2",
            "threshold": threshold,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec,
        })

        pd.DataFrame(results).to_csv(
            folder / "results.csv",
            index=False,
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

    if args.smoke:
        return

    frames = [pd.DataFrame(results)]

    if BASELINE.exists():
        baseline = pd.read_csv(BASELINE)
        baseline = baseline[
            baseline["category"].isin(args.categories)
        ].copy()
        baseline["upsampling"] = "bilinear"
        frames.append(baseline)
    else:
        print("주의: E5 기준 결과 파일이 없습니다.")

    comparison = pd.concat(
        frames, ignore_index=True
    )
    comparison.to_csv(
        OUT / "comparison.csv", index=False
    )

    print("\n===== E6 Upsampling 방식별 평균 =====")
    print(
        comparison.groupby("upsampling")[
            [
                "image_auroc",
                "f1",
                "pixel_aupro",
                "official_score",
            ]
        ].mean().round(4).to_string()
    )

    print("\n===== 카테고리별 종합 점수 =====")
    print(
        comparison.pivot_table(
            index="category",
            columns="upsampling",
            values="official_score",
        ).round(4).to_string()
    )


if __name__ == "__main__":
    main()
