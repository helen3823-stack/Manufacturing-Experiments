
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Dataset
from src.evaluate import infer, report
from src.metrics import f1
from example.sample_model import SampleModel

CATEGORIES = [
    "elec_01",
    "mech_05",
    "agri_02",
    "med_08",
    "groc_01",
]

OUT = ROOT / "artifacts" / "experiments" / "B0"
OUT.mkdir(parents=True, exist_ok=True)


def optimal_threshold(y, scores):
    """Validation F1을 최대화하는 임계값."""
    thresholds = np.unique(scores)
    best_threshold = float(thresholds[0])
    best_f1 = -1

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        value = f1(y, pred)["f1"]

        if value > best_f1:
            best_f1 = value
            best_threshold = float(thr)

    return best_threshold


def main():
    ds = Dataset()
    results = []

    for name in CATEGORIES:
        print(f"\n[{name}] B0 학습 시작", flush=True)

        cat = ds.category(name)

        t0 = time.time()
        model = SampleModel(
            cell=8,
            bank_size=20000,
            top=0.01,
            seed=42,
        )

        # Train 정상 이미지로만 학습
        model.fit(cat.images("train"))
        fit_time = time.time() - t0

        # Validation 전체 평가
        val = cat.val.copy()

        t1 = time.time()
        maps, scores = infer(
            model,
            cat.images(val),
            batch=8,
        )
        inference_time = time.time() - t1

        thr = optimal_threshold(
            val.is_anomaly.to_numpy(),
            scores,
        )

        val["score"] = scores
        val["pred"] = (scores >= thr).astype(int)

        masks = cat.masks(val)
        metrics = report(val, masks, maps)

        results.append({
            "category": name,
            "threshold": thr,
            **metrics,
            "fit_sec": fit_time,
            "inference_sec": inference_time,
        })

        print(
            f"AUROC={metrics['image_auroc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"AUPRO={metrics['pixel_aupro']:.4f}",
            flush=True,
        )

        pd.DataFrame(results).to_csv(
            OUT / "results.csv", index=False
        )

    df = pd.DataFrame(results)

    official = df[
        ["image_auroc", "f1", "pixel_aupro"]
    ].mean().mean()

    print("\n===== B0 결과 =====")
    print(df.to_string(index=False))
    print(f"\n대표 5개 평균 점수: {official:.4f}")


if __name__ == "__main__":
    main()
