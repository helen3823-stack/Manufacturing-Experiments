
import argparse
import time

import numpy as np
import pandas as pd
import torch

from run_b1 import (
    B1Model, CATEGORIES, ROOT,
    SEED, optimal_threshold
)
from src import Dataset
from src.evaluate import report

OUT = ROOT / "artifacts" / "experiments" / "E1"
BANK_SIZE = 10000
CANDIDATES = 30000
PROJ_DIM = 64


class E1Model(B1Model):
    @torch.inference_mode()
    def fit(self, cat, limit=None):
        rng = np.random.default_rng(SEED)

        # 1. 정상 패치 30k개 균등 무작위 수집
        selected = np.empty((0, 384), np.float32)
        selected_keys = np.empty(0, np.float64)

        for batch_idx, (_, images) in enumerate(
            cat.batches("train", batch=8, limit=limit)
        ):
            flat = self.features(images)
            flat = flat.reshape(-1, 384).cpu().numpy()

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
            f"  Candidate pool: {len(selected)}",
            flush=True
        )

        # 2. Random Projection: 384 -> 64
        generator = torch.Generator().manual_seed(SEED)
        projection = torch.randn(
            384, PROJ_DIM, generator=generator
        ) / (PROJ_DIM ** 0.5)

        projection = projection.to(self.device)

        candidates = torch.from_numpy(
            np.ascontiguousarray(selected)
        ).to(self.device)

        projected = candidates @ projection
        norms = projected.square().sum(dim=1)

        # 3. Greedy Farthest-First Selection
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
            k, dtype=torch.long,
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
            )
            dist = dist.clamp_min_(0)

            min_dist = torch.minimum(min_dist, dist)
            min_dist.masked_fill_(chosen, -1)

            if i + 1 < k:
                idx = min_dist.argmax()

            if (i + 1) % 2000 == 0:
                print(
                    f"  Greedy {i+1}/{k}",
                    flush=True
                )

        selected_idx = indices.cpu().numpy()
        bank = np.ascontiguousarray(
            selected[selected_idx]
        )

        # 4. 원래 384차원 특징을 Memory Bank에 저장
        self.bank = torch.from_numpy(bank).to(self.device)
        self.bank_sq = self.bank.square().sum(dim=1)

        print(
            f"  Final bank: {tuple(self.bank.shape)}",
            flush=True
        )

        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", nargs="+", default=CATEGORIES
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다.")

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    ds = Dataset()
    results = []

    print("Device:", device, flush=True)

    for name in args.categories:
        print(f"\n[{name}] E1 시작", flush=True)

        cat = ds.category(name)
        model = E1Model(device)

        start = time.time()
        model.fit(cat, limit=32 if args.smoke else None)
        torch.cuda.synchronize()
        fit_sec = time.time() - start

        if args.smoke:
            maps, scores = model.predict_batch(
                cat.images(cat.val.head(2))
            )
            print("Maps:", maps.shape)
            print("Scores:", scores.shape)
            print("Smoke Test 성공!")
            del model
            torch.cuda.empty_cache()
            continue

        val = cat.val.copy()
        all_maps = []
        all_scores = []

        start = time.time()

        for _, images in cat.batches(val, batch=4):
            maps, scores = model.predict_batch(images)
            all_maps.append(maps)
            all_scores.append(scores)

        torch.cuda.synchronize()
        infer_sec = time.time() - start

        maps = np.concatenate(all_maps)
        scores = np.concatenate(all_scores)

        threshold = optimal_threshold(
            val.is_anomaly.to_numpy(), scores
        )

        val["score"] = scores
        val["pred"] = (scores >= threshold).astype(int)

        masks = cat.masks(val)
        metrics = report(val, masks, maps)

        official = np.mean([
            metrics["image_auroc"],
            metrics["f1"],
            metrics["pixel_aupro"],
        ])

        model.save(OUT / name, threshold)

        results.append({
            "category": name,
            "threshold": threshold,
            **metrics,
            "official_score": official,
            "fit_sec": fit_sec,
            "inference_sec": infer_sec,
            "bank_size": len(model.bank),
        })

        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(
            OUT / "results.csv", index=False
        )

        print(
            f"  AUROC={metrics['image_auroc']:.4f}"
            f" F1={metrics['f1']:.4f}"
            f" AUPRO={metrics['pixel_aupro']:.4f}"
            f" Score={official:.4f}",
            flush=True
        )

        del model, maps, masks
        torch.cuda.empty_cache()

    if results:
        df = pd.DataFrame(results)
        score = df[
            ["image_auroc", "f1", "pixel_aupro"]
        ].mean().mean()

        print("\n===== E1 결과 =====")
        print(
            df[[
                "category", "official_score",
                "fit_sec", "inference_sec"
            ]].to_string(index=False)
        )
        print(f"\n대표 평균: {score:.4f}")


if __name__ == "__main__":
    main()
