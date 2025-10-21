import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("../outputs/csv/nba_players_advanced_2024_2025_30games.csv")

# 必要な列だけ抽出
df_subset = df[["PLAYER_ID", "PLAYER_NAME", "TS_PCT", "USG_PCT"]].dropna()

# スケーリング
scaler = StandardScaler()
scaled = scaler.fit_transform(df_subset[["TS_PCT", "USG_PCT"]])

# KMeansクラスタリング
kmeans = KMeans(n_clusters=4, random_state=42)
df_subset["Cluster"] = kmeans.fit_predict(scaled)

# 結果を確認
print(df_subset.groupby("Cluster")[["TS_PCT", "USG_PCT"]].mean())

# プロット
plt.figure(figsize=(8,6))
for c in df_subset["Cluster"].unique():
    cluster_data = df_subset[df_subset["Cluster"] == c]
    plt.scatter(cluster_data["USG_PCT"], cluster_data["TS_PCT"], label=f"Cluster {c}", alpha=0.7)

# 八村塁を赤でハイライト
rui = df_subset[df_subset["PLAYER_ID"] == 1629060]
if not rui.empty:
    plt.scatter(
        rui["USG_PCT"], rui["TS_PCT"],
        s=200, color="red", edgecolor="black", zorder=5, label="Rui Hachimura"
    )

plt.xlabel("Usage Rate (USG%)")
plt.ylabel("True Shooting % (TS%)")
plt.title("NBA Players Clustering by Usage and Efficiency (2024-25, GP >= 30)")
plt.legend()
plt.grid(True)

# プロットディレクトリを作成して保存
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(f"{plot_dir}/nba_scoring_efficiency_clustering.png", bbox_inches="tight", dpi=300)
