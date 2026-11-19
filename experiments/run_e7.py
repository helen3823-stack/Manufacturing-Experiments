
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

OUT = ROOT / "artifacts" / "experiments" / "E7"
BASELINE = (
    ROOT / "artifacts" / "experiments"
    / "E5" / "l2" / "results.csv"
)

SIGMAS = [0, 2, 4, 8]
SCORE_MODES = ["max", "top1", "top5"]


class E7Model(e5.E5Model):
    def __init__(self, device):
        super().__init__(device, mode="l2")

    @torch.inference_mode()
    def raw_patch_maps(self, images):
        # E5와 동일한 특징 추출 및 L2 normalization
        features = self.features(images)
        b, h, w, d = features.shape

        queries = features.reshape(-1, d)
        distances = []

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
        return anomaly.reshape(b, 1, h, w)


def image_scores(maps, mode):
    pixels = maps.reshape(maps.shape[0], -1)

    if mode == "max":
        return pixels.max(axis=1)

    ratio = {
        "top1": 0.01,
        "top5": 0.05
    }[mode]

    k = max(1, int(np.ceil(pixels.shape[1] * ratio)))

    return np.partition(
        pixels, -k, axis=1
    )[:, -k:].mean(axis=1)


def run_category(name, ds):
    print(f"\n[{name}] E7 시작", flush=True)

    cat = ds.category(name)
    device = torch.device("cuda")
    model = E7Model(device)

    torch.manual_seed(SEED)

    start = time.time()
    model.fit(cat)
    torch.cuda.synchronize()
    fit_sec = time.time() - start

    val = cat.val.copy()
    raw_list = []

    start = time.time()

    # 학습된 Memory Bank에서 32x32 거리 맵을 한 번 계산
    for _, images in cat.batches(val, batch=4):
        raw = model.raw_patch_maps(images)
        raw_list.append(raw.cpu())

    raw_maps = torch.cat(raw_list, dim=0)

    # E5와 동일한 Bilinear Upsampling
    upsampled = F.interpolate(
        raw_maps,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )[:, 0].numpy()

    torch.cuda.synchronize()
    infer_sec = time.time() - start

    # Validation GT mask: 평가에만 사용
    masks = cat.masks(val)
    labels = val.is_anomaly.to_numpy()

    results = []

    for sigma in SIGMAS:
        if sigma == 0:
            maps = upsampled
        else:
            maps = gaussian_filter(
                upsampled,
                sigma=(0, sigma, sigma)
            )

        for mode in SCORE_MODES:
            scores = image_scores(maps, mode)

            threshold = e5.e4.e3.e1.optimal_threshold(
                labels, scores
            )

            evaluated = val.copy()
            evaluated["score"] = scores
            evaluated["pred"] = (
                scores >= threshold
            ).astype(int)

            metrics = report(
                evaluated, masks, maps
            )

            official = float(np.mean([
                metrics["image_auroc"],
                metrics["f1"],
                metrics["pixel_aupro"]
            ]))

            results.append({
                "category": name,
                "sigma": sigma,
                "score_mode": mode,
                "threshold": threshold,
                "image_auroc": metrics["image_auroc"],
                "f1": metrics["f1"],
                "pixel_aupro": metrics["pixel_aupro"],
                "official_score": official,
                "fit_sec": fit_sec,
                "inference_sec": infer_sec
            })

            print(
                f"  σ={sigma} {mode}: "
                f"{official:.4f}",
                flush=True
            )

    model.close()
    del model
    torch.cuda.empty_cache()

    return pd.DataFrame(results)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="완료된 카테고리도 다시 실행"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    OUT.mkdir(parents=True, exist_ok=True)

    results_path = OUT / "results.csv"

    if results_path.exists() and not args.smoke:
        existing = pd.read_csv(results_path)
    else:
        existing = pd.DataFrame()

    ds = Dataset()

    for name in args.categories:
        if args.smoke:
            cat = ds.category(name)
            model = E7Model(torch.device("cuda"))
            model.fit(cat, limit=32)

            raw = model.raw_patch_maps(
                cat.images(cat.val.head(2))
            )

            print(
                f"Smoke OK: {tuple(raw.shape)}"
            )

            model.close()
            del model
            torch.cuda.empty_cache()
            continue

        # 이미 12개 설정 완료된 카테고리는 건너뛰기
        if not args.force and not existing.empty:
            prev = existing[
                existing["category"] == name
            ]

            if len(prev) == 12:
                print(
                    f"[{name}] 이미 완료 — Skip"
                )
                continue

        df = run_category(name, ds)

        if not existing.empty:
            existing = existing[
                existing["category"] != name
            ]

        existing = pd.concat(
            [existing, df], ignore_index=True
        )

        existing.to_csv(
            results_path, index=False
        )

        print(
            f"[{name}] 12개 설정 저장 완료",
            flush=True
        )

    if args.smoke:
        return

    if existing.empty:
        print("결과가 없습니다.")
        return

    summary = (
        existing.groupby(["sigma", "score_mode"])
        .agg(
            n_categories=("category", "nunique"),
            image_auroc=("image_auroc", "mean"),
            f1=("f1", "mean"),
            pixel_aupro=("pixel_aupro", "mean"),
            official_score=("official_score", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "comparison.csv", index=False
    )

    # 전체 대표 카테고리가 완료된 설정만 순위 비교
    complete = summary[
        summary["n_categories"] == len(args.categories)
    ].sort_values(
        "official_score", ascending=False
    )

    print("\n===== E7 조합별 성능 =====")
    print(
        summary.round(4).to_string(index=False)
    )

    if not complete.empty:
        print("\n===== E7 최고 설정 =====")
        print(
            complete.head(5).round(4).to_string(
                index=False
            )
        )

        best = complete.iloc[0]
        print(
            f"\nBest: σ={int(best['sigma'])}, "
            f"score={best['score_mode']}, "
            f"평균={best['official_score']:.4f}"
        )

    # E5 결과와 기존 설정(σ=4, Top1%) 일치 확인
    if BASELINE.exists():
        base = pd.read_csv(BASELINE)
        base = base.set_index("category")

        current = existing[
            (existing["sigma"] == 4) &
            (existing["score_mode"] == "top1")
        ].set_index("category")

        common = base.index.intersection(current.index)

        if len(common) > 0:
            diff = (
                base.loc[common, "official_score"]
                - current.loc[common, "official_score"]
            ).abs()

            print(
                "\nE5 기준 재현 최대 차이:",
                round(float(diff.max()), 6)
            )

            if diff.max() > 0.001:
                print(
                    "WARNING: E5 기준 결과와 차이가 큽니다. "
                    "실험 설정을 확인하세요."
                )


if __name__ == "__main__":
    main()
