import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load CSV
csv_path = r'Parameters_Summary\merged_Summary.csv'  # <-- Change this to your file path
df = pd.read_csv(csv_path)

# Step 2: Prepare data grouped by 'bandname'
band_groups = df.groupby('band_name')['max_val'].apply(list)

# Step 3: Plot boxplot
fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot, without outliers (showfliers=False)
box = ax.boxplot(band_groups.tolist(),
                 patch_artist=True,
                 showfliers=False,
                 boxprops=dict(color='gray'),
                 whiskerprops=dict(color='gray'),
                 capprops=dict(color='gray'),
                 medianprops=dict(color='gray'),
                 flierprops=dict(markerfacecolor='red', marker='o', markersize=3))

# Set x-axis labels to band names
ax.set_xticklabels(band_groups.index, rotation=90)
ax.set_title('Max Value Distribution by Bandname')
ax.set_xlabel('Bandname')
ax.set_ylabel('Max Value')

# Make boxes thinner
for patch in box['boxes']:
    patch.set_linewidth(0.8)

plt.tight_layout()
plt.savefig('Max_Summary.jpg', dpi=300)

# Step 2: Prepare data grouped by 'band_name' and calculate required percentiles
percentiles = [0.1, 5, 90, 95, 99, 99.9]
result = df.groupby('band_name')['max_val'].quantile([p/100 for p in percentiles]).unstack()

# Rename columns to indicate percentiles
result.columns = [f'percentile_{p}' for p in percentiles]

# Step 3: Save to new CSV
output_path = r'Parameters_Summary\percentile_summary.csv'  # <-- Change if needed
result.to_csv(output_path)

print(f"Percentile summary saved to {output_path}")