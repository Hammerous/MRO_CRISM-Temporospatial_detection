import pandas as pd
from scipy.stats import pearsonr

# 读取输入数据
df = pd.read_csv('mineral_summary_modified.csv')

# 准备存储结果的列表
results = []

# 按 band_name 分组
for band, group in df.groupby('band_name'):
    # 计算 mean_val 与 CenterLat 的相关性
    if group['mean_val'].nunique() > 1 and group['CenterLat'].nunique() > 1:
        r_lat, p_lat = pearsonr(group['mean_val'], group['CenterLat'])
    else:
        r_lat, p_lat = float('nan'), float('nan')
    
    # 计算 mean_val 与 SolLong 的相关性
    if group['mean_val'].nunique() > 1 and group['SolLong'].nunique() > 1:
        r_sl, p_sl = pearsonr(group['mean_val'], group['SolLong'])
    else:
        r_sl, p_sl = float('nan'), float('nan')
    
    results.append({
        'band_name': band,
        'r_CenterLat': r_lat,
        'p_CenterLat': p_lat,
        'r_SolLong': r_sl,
        'p_SolLong': p_sl
    })

# 转换为 DataFrame 并保存
result_df = pd.DataFrame(results)
result_df.to_csv('correlation_results.csv', index=False)

print("Correlation analysis complete. Results saved to 'correlation_results.csv'.")
