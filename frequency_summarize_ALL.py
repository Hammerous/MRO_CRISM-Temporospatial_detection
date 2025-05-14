"""
Summarise mineral-distribution statistics across multiple observation CSVs.

Steps
-----
1. Iterate over all *.csv files in `SOURCE_DIR`.
2. For each file:
   a. Group records by 'band_name'.
   b. Discard groups containing only one record (cannot describe change).
   c. For the remaining groups, compute group means of
      ['count', 'mean_val', 'median_val', 'std_val'].
   d. For every original record in the retained groups, calculate its
      value as a percentage of the corresponding group mean.
   e. Keep only the percentage columns plus 'ProductId'.
3. Aggregate the per-file results and write a single summary CSV.
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# User parameters                                                             #
# --------------------------------------------------------------------------- #
SOURCE_DIR   = Path(r"./Change_Detect")      # folder containing the raw CSVs
OUTPUT_FILE  = Path(r"./mineral_summary.csv")  # destination for the summary

# --------------------------------------------------------------------------- #
# Core processing                                                             #
# --------------------------------------------------------------------------- #
def summarise_single_csv(csv_path: Path) -> pd.DataFrame:
    print(csv_path)
    """Return a DataFrame containing percentage statistics for one CSV."""
    df = pd.read_csv(csv_path)

    metrics = ["mean_val", "median_val", "std_val"]
    df = df[~((df[metrics] == 0).all(axis=1) & (df["count"] != 0))]
    ### deal with missing RPEAK1 records


    # Identify numeric columns of interest
    metrics = ["count", "mean_val", "median_val", "std_val"]

    # Split into groups by band_name
    grouped = df.groupby("band_name", group_keys=False)

    result_frames = []

    for band, grp in grouped:
        if grp.shape[0] < 3:          # ‑‑ skip csvs without temporal variation more than two views
            return pd.DataFrame()

        # Compute group means once
        means = grp[metrics].mean()

        # Express each metric as percentage of the group mean
        pct_frame = grp.copy()
        for col in metrics:
            pct_frame[f"{col}_pct"] = (100 * pct_frame[col] / means[col])

        # Keep only what the specification asks for
        keep_cols = ["ProductId", "band_name", "CenterLat","CenterLon","EmAngle","InAngle","PhAngle","SolLong","UTCstart"] + [f"{m}_pct" for m in metrics] + metrics
        result_frames.append(pct_frame[keep_cols])

    # Concatenate results for all bands belonging to this file
    if result_frames:
        result_df = pd.concat(result_frames, ignore_index=True)
        # print(len(pd.unique(df['band_name'])))
        # print(len(pd.unique(result_df['band_name'])))
        # print(set(pd.unique(df['band_name'])) - set(pd.unique(result_df['band_name'])))
        return result_df
    else:
        return pd.DataFrame()

# --------------------------------------------------------------------------- #
# Run for every CSV in the source folder                                      #
# --------------------------------------------------------------------------- #
all_results = []

for csv_file in SOURCE_DIR.glob("*.csv"):
    summary = summarise_single_csv(csv_file)
    if not summary.empty:
        # Optionally annotate origin file
        summary.insert(0, "source_csv", csv_file.name)
        all_results.append(summary)

# --------------------------------------------------------------------------- #
# Concatenate and write to disk                                               #
# --------------------------------------------------------------------------- #
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Summary written to {OUTPUT_FILE.resolve()}")
else:
    print("No qualifying groups found in any CSV.")
