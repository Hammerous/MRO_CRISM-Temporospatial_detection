import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from adjustText import adjust_text  # pip install adjustText

def main():
    # -------------------------------------------------------------------------
    # 1. Load data (ensure UTF-8 for Chinese)
    # -------------------------------------------------------------------------
    csv_path = "mineral_variability_RRI.csv"  # <-- Update this path
    df = pd.read_csv(csv_path, encoding='utf-8')

    # -------------------------------------------------------------------------
    # 2. Extract features
    # -------------------------------------------------------------------------
    features = df[['RRI', 'cov_real_mean']].values

    # -------------------------------------------------------------------------
    # 3. (Optional) Determine optimal cluster count via silhouette score
    # -------------------------------------------------------------------------
    best_k = 3
    best_score = -1
    for k in range(2, 6):
        km_tmp = KMeans(n_clusters=k, random_state=42).fit(features)
        score = silhouette_score(features, km_tmp.labels_)
        if score > best_score:
            best_score, best_k = score, k

    print(f"Optimal number of clusters: {best_k}")

    # -------------------------------------------------------------------------
    # 4. Apply K‑Means
    # -------------------------------------------------------------------------
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)

    # -------------------------------------------------------------------------
    # 5. Plot with Chinese font support
    # -------------------------------------------------------------------------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # Windows 常见中文字体
    # plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

    plt.figure(figsize=(10, 6))
    for cid in sorted(df['cluster'].unique()):
        sub = df[df['cluster'] == cid]
        plt.scatter(sub['RRI'], sub['cov_real_mean'], label=f'Cluster {cid}', s=50)

    # Add non‑overlapping labels
    texts = []
    for _, row in df.iterrows():
        texts.append(
            plt.text(
                row['RRI'], row['cov_real_mean'],
                row['band_name'],
                fontsize=8,
                ha='center', va='center'
            )
        )
    adjust_text(
        texts,
        only_move={'points':'y', 'texts':'y'},
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
    )

    # Axis scaling & labels
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("时间")
    plt.ylabel("空间")
    plt.title("K-Means Clustering of Bands")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
