import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === データ ===
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_40games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)
# === 特徴量（スコアラー分析用） ===
features = ["GP", "MIN", "PTS", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT"]
X = df[features].fillna(0)
# === クラスタリング ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(X_scaled)
# === 可視化（FG3A × FG3_PCT） ===
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df, x="FG3A", y="FG3_PCT",
    hue="Cluster", palette="tab10", alpha=0.8
)
# 八村塁を赤でハイライト
rui = df[df["PLAYER_ID"] == 1629060]
plt.scatter(
    rui["FG3A"], rui["FG3_PCT"],
    s=200, color="red", edgecolor="black", zorder=5, label="Rui Hachimura"
)
plt.text(
    rui["FG3A"].values[0] + 0.5,
    rui["FG3_PCT"].values[0],
    "Rui Hachimura",
    color="red", fontsize=10, weight="bold"
)
plt.title("NBA Scorer Map (FG3A × FG3_PCT) — 3-point attempts & efficiency", fontsize=14)
plt.xlabel("3-Point Field Goal Attempts (FG3A)")
plt.ylabel("3-Point Field Goal Percentage (FG3_PCT)")
plt.legend()
plt.grid(alpha=0.3)

path = f"{plot_dir}/nba_scorer_map_fg3a_fg3_pct_rui.png"
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.close()

print(f"スコアラー分析グラフ出力完了: {path}")
