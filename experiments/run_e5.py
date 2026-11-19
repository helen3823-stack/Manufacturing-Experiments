
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import run_e4 as e4
from run_b1 import ROOT, CATEGORIES, SEED
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E5"

BASELINE = (
    ROOT / "artifacts" / "experiments"
    / "E4" / "neighborhood_5" / "results.csv"
)

MODES = ["l2", "standard"]


class E5Model(e4.E4Model):
    def __init__(self, device, mode):
        super().__init__(device, neighborhood=5)
        self.mode = mode
        self.feature_mean = None
        self.feature_std = None

    @torch.inference_mode()
    def features(self, images):
        x = super().features(images)

        if self.mode == "l2":
            return F.normalize(x, p=2, dim=-1)

        if self.mode == "standard":
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("Train 통계가 먼저 계산되어야 합니다.")
            return (x - self.feature_mean) / self.feature_std

        return x

    @torch.inference_mode()
    def calculate_train_stats(self, cat, limit=None):
        # 정상 Train만 사용하여 통계 계산
        total = None
        total_sq = None
        count = 0

        for _, images in cat.batches(
            "train", batch=8, limit=limit
        ):
            # 정규화 적용 전 E4 특징
            x = e4.E4Model.features(self, images)
            x = x.reshape(-1, self.feature_dim)
            x = x.to(dtype=torch.float64)

            batch_sum = x.sum(dim=0)
            batch_sq = x.square().sum(dim=0)

            if total is None:
                total = batch_sum
                total_sq = batch_sq
            else:
                total += batch_sum
                total_sq += batch_sq

            count += x.shape[0]

        if count == 0:
            raise RuntimeError("Train 패치가 없습니다.")

        mean = total / count
        var = total_sq / count - mean.square()

        self.feature_mean = mean.float().reshape(1, 1, 1, -1)
        self.feature_std = var.clamp_min(1e-12).sqrt().float().reshape(1, 1, 1, -1)

        print(
            f"  Train statistics: {count} patches",
            flush=True
        )

    def fit(self, cat, limit=None):
        if self.mode == "standard":
            self.calculate_train_stats(cat, limit=limit)

        # E3의 Greedy Coreset 학습을 그대로 사용
        return super().fit(cat, limit=limit)

    def save_normalization(self, folder):
        folder.mkdir(parents=True, exist_ok=True)

        if self.mode == "standard":
            np.savez_compressed(
                folder / "normalization.npz",
                mean=self.feature_mean.cpu().numpy(),
                std=self.feature_std.cpu().numpy(),
                mode=self.mode,
            )


def run_one(mode, categories, smoke=False):
    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    ds = Dataset()
    results = []

    for name in categories:
        print(
            f"\n[{name}] E5 {mode} 시작",
            flush=True
        )

        cat = ds.category(name)
        model = E5Model(device, mode)

        start = time.time()
        model.fit(
            cat, limit=32 if smoke else None
        )
        torch.cuda.synchronize()
        fit_sec = time.time() - start

        if smoke:
            maps, scores = model.predict_batch(
                cat.images(cat.val.head(2))
            )
            print(
                "Smoke OK:",
                maps.shape,
                scores.shape,
                flush=True
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

        threshold = e4.e3.e1.optimal_threshold(
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

        official = float(np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"],
        ]))

        model_folder = folder / name
        model.save(model_folder, threshold)
        model.save_normalization(model_folder)

        results.append({
            "category": name,
            "normalization": mode,
            "bank_size": 5000,
            "neighborhood": 5,
            "threshold": threshold,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec,
        })

        # 카테고리별 결과 즉시 저장
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    torch.manual_seed(SEED)

    for mode in args.modes:
        run_one(mode, args.categories, args.smoke)

    if args.smoke:
        return

    frames = []

    if BASELINE.exists():
        base = pd.read_csv(BASELINE)
        base = base[
            base["category"].isin(args.categories)
        ].copy()
        base["normalization"] = "raw"
        frames.append(base)
    else:
        print("주의: E4 Raw 기준 결과가 없습니다.")

    for mode in MODES:
        path = OUT / mode / "results.csv"

        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[
                frame["category"].isin(args.categories)
            ]
            frames.append(frame)

    if not frames:
        return

    combined = pd.concat(
        frames, ignore_index=True
    )
    combined.to_csv(
        OUT / "comparison.csv",
        index=False
    )

    print("\n===== E5 정규화 방식별 평균 =====")
    print(
        combined.groupby("normalization")[
            [
                "image_auroc",
                "f1",
                "pixel_aupro",
                "official_score"
            ]
        ].mean().round(4).to_string()
    )

    print("\n===== 카테고리별 종합 점수 =====")
    print(
        combined.pivot_table(
            index="category",
            columns="normalization",
            values="official_score"
        ).round(4).to_string()
    )


if __name__ == "__main__":
    main()
