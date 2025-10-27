import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)

# 3ポイント関連の特徴量のみを使用
features = ["FG3A", "FG3_PCT"]

# 欠損値を0で埋める
df_clean = df[features].fillna(0)

# クラスタリング
X = df_clean.values
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# === クラスタラベル再定義 ===
cluster_labels, cluster_colors = {}, {}

for i in range(3):
    cdata = df[df["Cluster"] == i]
    avg_attempts = cdata["FG3A"].mean()
    avg_percentage = cdata["FG3_PCT"].mean()

    if avg_attempts > 5 and avg_percentage > 0.35:
        cluster_labels[i] = "Elite 3PT Shooter"
        cluster_colors[i] = "red"
    elif avg_attempts > 5:
        cluster_labels[i] = "High Volume 3PT"
        cluster_colors[i] = "blue"
    elif avg_percentage > 0.35:
        cluster_labels[i] = "Efficient 3PT"
        cluster_colors[i] = "green"
    else:
        cluster_labels[i] = "Limited 3PT"
        cluster_colors[i] = "gray"

df["Playstyle"] = df["Cluster"].map(cluster_labels)

# 可視化
plt.figure(figsize=(12, 8))
for i in range(3):
    cdata = df[df["Cluster"] == i]
    plt.scatter(
        cdata["FG3A"], cdata["FG3_PCT"],
        c=cluster_colors[i], label=cluster_labels[i],
        s=70, alpha=0.6
    )

# 八村塁ハイライト
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if not rui.empty:
    plt.scatter(
        rui["FG3A"], rui["FG3_PCT"],
        s=250, color="gold", edgecolor="black", zorder=5, label="Rui Hachimura"
    )
    plt.text(
        rui["FG3A"].values[0] + 0.2,
        rui["FG3_PCT"].values[0],
        "Rui Hachimura", color="black", fontsize=11, weight="bold"
    )

plt.title("NBA Player 3-Point Shooting Clustering", fontsize=15)
plt.xlabel("3-Point Attempts per Game (FG3A)")
plt.ylabel("3-Point Percentage (FG3_PCT)")
plt.legend()
plt.grid(alpha=0.3)

path = f"{plot_dir}/nba_3pt_clustering.png"
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.close()

# 結果出力
print("3ポイントシュートクラスタ分析結果")
for i in range(3):
    cdata = df[df["Cluster"] == i]
    print(f"\n{cluster_labels[i]} ({len(cdata)}人):")
    print(f"  平均3P試行数: {cdata['FG3A'].mean():.3f}")
    print(f"  平均3P成功率: {cdata['FG3_PCT'].mean():.3f}")
    print(f"  主な選手例: {cdata.nlargest(3, 'MIN')['PLAYER_NAME'].tolist()}")

if not rui.empty:
    print(f"\n八村塁の分類: {rui['Playstyle'].values[0]}")
    print(f"八村塁の3P試行数: {rui['FG3A'].values[0]:.1f}")
    print(f"八村塁の3P成功率: {rui['FG3_PCT'].values[0]:.3f}")

print(f"\n出力完了: {path}")
