import pandas as pd
import matplotlib.pyplot as plt
import os

# === Step 1: Load the bin interval reference CSV ===
# Example path: 'bin_intervals.csv'
bin_interval_path = 'pairs_60x100.csv'
bin_intervals = pd.read_csv(bin_interval_path, index_col=0)

# === Step 2: Load the data CSV and group by "band_name" ===
# Example path: 'data_values.csv'
data_path = r'Change_Detect\3-297.csv'
data_df = pd.read_csv(data_path)
grouped = data_df.groupby("band_name")

# === Step 5: Create output folder ===
out_dir = os.path.splitext(os.path.basename(data_path))[0]
os.makedirs(out_dir, exist_ok=True)

# === Step 4: Plot the value distributions for each group ===
bin_cols = [f'bin_{i}' for i in range(1, 101)]

# ── 5. loop over each mineral (band_name) ─────────────────────────────────────
for mineral, g in grouped:
    # skip minerals missing bin‑edge definition
    if mineral not in bin_intervals.index:
        print(f'[WARN] {mineral} missing in bin-edge table — skipped')
        continue

    # x‑axis: real bin centres/edges for this mineral, as numpy array
    x = bin_intervals.loc[mineral, bin_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    # iterate over every observation (row) in this mineral‑specific group
    for _, row in g.iterrows():
        y = row[bin_cols].to_numpy(dtype=float)          # pixel‑count distribution
        pid = row.get('ProductId', 'unknown')
        ax.plot(x, y, alpha=0.6, linewidth=1.2, label = f"{pid} (n={int(y.sum())})")

    # aesthetics
    ax.set_title(f'Pixel-intensity distribution - {mineral}')
    ax.set_xlabel('Intensity bin')
    ax.set_ylabel('Pixel count')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # show legend only when ≤15 traces; otherwise omit to avoid clutter
    if len(g) <= 15:
        ax.legend(title='ProductId', fontsize='small', frameon=False)

    # save figure
    fig_path = os.path.join(out_dir, f'{mineral}.png')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

print(f'All mineral plots saved in: {out_dir}')
