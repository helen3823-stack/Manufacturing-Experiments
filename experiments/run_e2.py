
import argparse
import time

import numpy as np
import pandas as pd
import torch

import run_e1 as e1

from run_b1 import CATEGORIES, ROOT, SEED
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E2"
E1_RESULTS = ROOT / "artifacts" / "experiments" / "E1" / "results.csv"


def run_experiment(bank_size, categories):
    # E1에서 사용한 Greedy 방식 그대로 유지
    # 선택할 Memory Bank 개수만 변경
    e1.BANK_SIZE = bank_size

    device = torch.device("cuda")
    ds = Dataset()
    results = []

    folder = OUT / f"bank_{bank_size}"
    folder.mkdir(parents=True, exist_ok=True)

    for name in categories:
        print(
            f"\n[{name}] E2 Bank {bank_size} 시작",
            flush=True
        )

        cat = ds.category(name)
        model = e1.E1Model(device)

        t0 = time.time()
        model.fit(cat)

        torch.cuda.synchronize()
        fit_sec = time.time() - t0

        # Validation 평가
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

        # 카테고리별 Val F1 최적 임계값
        threshold = e1.optimal_threshold(
            val.is_anomaly.to_numpy(),
            scores
        )

        val["score"] = scores
        val["pred"] = (scores >= threshold).astype(int)

        masks = cat.masks(val)
        metrics = report(val, masks, maps)

        official = np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"]
        ])

        # 각 카테고리 모델 별도 저장
        model.save(folder / name, threshold)

        results.append({
            "category": name,
            "bank_size": bank_size,
            "threshold": threshold,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec
        })

        # 카테고리 하나가 끝날 때마다 저장
        pd.DataFrame(results).to_csv(
            folder / "results.csv",
            index=False
        )

        print(
            f"AUROC={metrics['image_auroc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"AUPRO={metrics['pixel_aupro']:.4f} "
            f"Score={official:.4f}",
            flush=True
        )

        del model, maps, masks
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    avg = df[
        ["image_auroc", "f1", "pixel_aupro"]
    ].mean().mean()

    print(
        f"\nBank {bank_size} 평균: {avg:.4f}",
        flush=True
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5000, 20000]
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다.")

    torch.manual_seed(SEED)

    all_results = []

    # 기존 E1 10k 결과 불러오기
    if E1_RESULTS.exists():
        baseline = pd.read_csv(E1_RESULTS)
        baseline = baseline[
            baseline["category"].isin(args.categories)
        ].copy()
        baseline["bank_size"] = 10000
        all_results.append(baseline)

    for size in args.sizes:
        df = run_experiment(size, args.categories)
        all_results.append(df)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        OUT / "comparison.csv",
        index=False
    )

    print("\n===== E2 비교 결과 =====")

    summary = (
        combined.groupby("bank_size")
        .agg(
            n_categories=("category", "nunique"),
            image_auroc=("image_auroc", "mean"),
            f1=("f1", "mean"),
            pixel_aupro=("pixel_aupro", "mean"),
            fit_sec=("fit_sec", "mean"),
            inference_sec=("inference_sec", "mean")
        )
    )

    summary["official_score"] = summary[
        ["image_auroc", "f1", "pixel_aupro"]
    ].mean(axis=1)

    print(summary.round(4).to_string())

    print("\n카테고리별 종합 점수")
    print(
        combined.pivot_table(
            index="category",
            columns="bank_size",
            values="official_score"
        ).round(4).to_string()
    )


if __name__ == "__main__":
    main()
