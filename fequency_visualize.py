"""
Assess variability of mineral‑distribution indices.

Input  : mineral_summary.csv   (created by the first script)
Output : mineral_variability.csv  (per‑band standard deviations)
"""

from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------- #
# Paths                                                                #
# -------------------------------------------------------------------- #
SOURCE_FILE = Path(r"./mineral_summary.csv")
OUTPUT_FILE1 = Path(r"./mineral_variability.csv")
OUTPUT_FILE2 = Path(r"./mineral_time_intervals.csv")

# -------------------------------------------------------------------- #
# Load and check                                                        #
# -------------------------------------------------------------------- #
df = pd.read_csv(SOURCE_FILE)

# 1. Define which columns are truly “percent” metrics
pct_metrics = ["count_pct", "mean_val_pct", "median_val_pct", "std_val_pct"]

# 2. Group and aggregate:
#    - For each percent metric, compute its std → "{metric}_std"
#    - For mean_val, compute both its mean and its std
var_df = (
    df
    .groupby("band_name")
    .agg(
        **{f"{m}_std": (m, "std") for m in pct_metrics},
        mean_val_std = ("mean_val", "std"),
        mean_val     = ("mean_val", "mean")
    )
    .reset_index()
)

# 3. Compute coefficient of variation (std/mean), replacing any zero means with NA
var_df["cov_real_mean"] = (
    var_df["mean_val_std"] 
    / var_df["mean_val"].replace(0, pd.NA)
)

# df['response_sum'] = df['mean_val'] *df['count']

# 2. Group and aggregate:
#    - For each percent metric, compute its std → "{metric}_std"
#    - For mean_val, compute both its mean and its std
# var_df = (
#     df
#     .groupby("band_name")
#     .agg(
#         **{f"{m}_std": (m, "std") for m in pct_metrics},
#         response_sum_std = ("response_sum", "std"),
#         response_sum = ("response_sum", "mean"),
#         mean_val_std = ("mean_val", "std"),
#         mean_val     = ("mean_val", "mean")
#     )
#     .reset_index()
# )

# # 3. Compute coefficient of variation (std/mean), replacing any zero means with NA
# var_df["cov_response_sum"] = (
#     var_df["response_sum_std"] 
#     / var_df["response_sum"].replace(0, pd.NA)
# )

var_df.to_csv(OUTPUT_FILE1, index=False)
print(f"✓ Variability statistics written to {OUTPUT_FILE1.resolve()}")

# Parse UTCStart as datetime
df["UTCstart"] = pd.to_datetime(df["UTCstart"], errors="coerce")
if df["UTCstart"].isna().any():
    raise ValueError("Some UTsStart values could not be parsed as datetimes.")

# -------------------------------------------------------------------- #
# Compute per‐group intervals (in days)                                #
# -------------------------------------------------------------------- #
records = []
for roi_id, grp in df.groupby("source_csv"):
    # sort ascending
    times = grp.sort_values("UTCstart")["UTCstart"].drop_duplicates()
    
    # successive differences → Timedelta, then convert to days
    day_deltas = times.diff().dt.total_seconds() / 86400.0
    day_deltas = day_deltas.dropna()  # now length ≥ 2 since ≥3 records per group
    
    # aggregate
    records.append({
        "source_csv": roi_id,
        "min_interval_days": day_deltas.min(),
        "avg_interval_days": day_deltas.mean(),
        "max_interval_days": day_deltas.max(),
    })

out_df = pd.DataFrame.from_records(records)

# -------------------------------------------------------------------- #
# Write results                                                        #
# -------------------------------------------------------------------- #
out_df.to_csv(OUTPUT_FILE2, index=False)
print(f"✓ Wrote time-intervals (in days) to {OUTPUT_FILE2.resolve()}")

# ---------------------- #
# Group and summarize    #
# ---------------------- #
import json
features = []

for roi_id, grp in df.groupby("source_csv"):
    avg_lat = grp["CenterLat"].mean()
    avg_lon = grp["CenterLon"].mean()
    count = grp.shape[0]

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [avg_lon, avg_lat]
        },
        "properties": {
            "source_csv": roi_id,
            "count": count
        }
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# ---------------------- #
# Write GeoJSON output   #
# ---------------------- #
with open("OUTPUT_GEOJSON.json", "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2, ensure_ascii=False)

print(f"✓ Wrote GeoJSON file")