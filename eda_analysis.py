
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "dataset" / "index.csv")

out = ROOT / "artifacts" / "eda"
out.mkdir(parents=True, exist_ok=True)

# 1. 카테고리별 데이터 수
df["kind"] = np.where(
    df["split"] == "train", "train",
    np.where(df["is_anomaly"] == 1, "val_bad", "val_good")
)

counts = (
    df.groupby(["group", "category", "kind"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=["train", "val_good", "val_bad"], fill_value=0)
      .reset_index()
)

# 2. 결함 면적 분석
bad = df[(df["split"] == "val") & (df["is_anomaly"] == 1)].copy()

bad["size"] = pd.cut(
    bad["mask_area_pct"],
    bins=[-np.inf, 1, 5, np.inf],
    labels=["small", "medium", "large"],
    right=False
)

area = (
    bad.groupby("category", observed=True)
       .agg(
           median_area=("mask_area_pct", "median"),
           mean_area=("mask_area_pct", "mean"),
           min_area=("mask_area_pct", "min"),
           max_area=("mask_area_pct", "max"),
       )
)

size_counts = pd.crosstab(bad["category"], bad["size"])
size_counts = size_counts.reindex(
    columns=["small", "medium", "large"], fill_value=0
)
size_counts["small_ratio"] = (
    size_counts["small"] / size_counts.sum(axis=1)
)

# 3. 전체 요약 테이블
summary = counts.merge(
    area, left_on="category", right_index=True, how="left"
).merge(
    size_counts, left_on="category", right_index=True, how="left"
)

summary = summary.sort_values(["group", "category"])
summary.to_csv(out / "category_summary.csv", index=False)

# 4. 결함 유형별 개수
defect_types = (
    bad.groupby(["group", "category", "label"])
       .size()
       .reset_index(name="count")
)
defect_types.to_csv(out / "defect_types.csv", index=False)

print(f"카테고리 수: {df['category'].nunique()}")
print(f"전체 이미지 수: {len(df):,}")
print(f"Train 이미지 수: {(df['split']=='train').sum():,}")
print(f"Val 불량 이미지 수: {len(bad):,}")
print("\n작은 결함 비율 TOP 10")
print(
    summary[["category", "val_bad", "median_area", "small_ratio"]]
    .sort_values("small_ratio", ascending=False)
    .head(10)
    .to_string(index=False)
)
print(f"\n결과 저장 완료: {out}")
