import os
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_advanced_2024_2025_30games_clustering.csv")
df_subset = df[["PLAYER_ID", "PLAYER_NAME", "TS_PCT", "USG_PCT"]].dropna().copy()
# USGを%に変換
if df_subset["USG_PCT"].max() <= 1:
    df_subset["USG_PCT"] = df_subset["USG_PCT"] * 100
# TSを%に変換
if df_subset["TS_PCT"].max() <= 1:
    df_subset["TS_PCT"] = df_subset["TS_PCT"] * 100
# 分割
low_usg = df_subset[df_subset["USG_PCT"] < 20].copy()
high_usg = df_subset[df_subset["USG_PCT"] >= 20].copy()
# 低USGクラスタ分割
if len(low_usg) > 0:
    q1 = low_usg["TS_PCT"].quantile(0.33)
    q2 = low_usg["TS_PCT"].quantile(0.66)
    low_usg.loc[low_usg["TS_PCT"] < q1, "Cluster"] = 0
    low_usg.loc[(low_usg["TS_PCT"] >= q1) & (low_usg["TS_PCT"] < q2), "Cluster"] = 1
    low_usg.loc[low_usg["TS_PCT"] >= q2, "Cluster"] = 2

# 高USGは一律Cluster=3
if len(high_usg) > 0:
    high_usg["Cluster"] = 3
# 明示的に型を揃えて結合
low_usg["Cluster"] = low_usg["Cluster"].astype(int)
high_usg["Cluster"] = high_usg["Cluster"].astype(int)
df_subset = pd.concat([low_usg, high_usg], ignore_index=True)
# ラベル定義
cluster_labels = {
    0: "Low USG / Low Efficiency",
    1: "Low USG / Mid Efficiency",
    2: "Low USG / High Efficiency",
    3: "High USG Scorers",
}

cluster_colors = {
    0: "purple",
    1: "blue",
    2: "green",
    3: "red",
}

# ラベル追加
df_subset["Player_Role"] = df_subset["Cluster"].map(cluster_labels)
# プロット
plt.figure(figsize=(12, 8))
for i in range(4):
    cluster_data = df_subset[df_subset["Cluster"] == i]
    plt.scatter(
        cluster_data["USG_PCT"], cluster_data["TS_PCT"],
        c=cluster_colors[i], label=cluster_labels[i],
        alpha=0.75, s=70, edgecolor="white", linewidth=0.5
    )

# USG 20%の境界線
plt.axvline(x=20, color='gray', linestyle='--', alpha=0.6, label="Usage Rate 20%")
# 八村を強調
rui = df_subset[df_subset["PLAYER_ID"] == 1629060]
if not rui.empty:
    plt.scatter(
        rui["USG_PCT"], rui["TS_PCT"],
        s=250, color="gold", edgecolor="black", zorder=5, label="Rui Hachimura"
    )
    plt.text(
        rui["USG_PCT"].values[0] + 0.6,
        rui["TS_PCT"].values[0],
        "Rui Hachimura", color="black", fontsize=11, weight="bold"
    )

plt.xlabel("Usage Rate (%)", fontsize=12)
plt.ylabel("True Shooting (%)", fontsize=12)
plt.title("NBA Players Role Segmentation", fontsize=14, weight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)

plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(f"{plot_dir}/nba_role_segmentation.png", bbox_inches="tight", dpi=300)
print(f"\n出力完了: {plot_dir}/nba_role_segmentation.png")
