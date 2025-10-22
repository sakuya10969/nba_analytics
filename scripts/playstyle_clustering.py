import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)

# 特徴量
offense_features = ["PTS", "AST", "FG_PCT", "FG3_PCT", "FT_PCT", "FGM", "TOV"]
defense_features = ["STL", "BLK", "DREB", "REB"]

# 攻守別で標準化（これが重要ポイント）
df_scaled = df.copy()

off_scaler = StandardScaler()
def_scaler = StandardScaler()

df_scaled[offense_features] = off_scaler.fit_transform(df[offense_features].fillna(0))
df_scaled[defense_features] = def_scaler.fit_transform(df[defense_features].fillna(0))

# 攻撃スコア
df["Offense_Score"] = (
    df_scaled["PTS"] * 0.35 +      # 得点量
    df_scaled["AST"] * 0.25 +      # アシスト貢献
    df_scaled["FGM"] * 0.15 +      # 攻撃量
    df_scaled["FG_PCT"] * 0.15 +   # 効率
    df_scaled["FT_PCT"] * 0.10 -   # フリースロー効率
    df_scaled["TOV"] * 0.10        # 攻撃ロス
)

# 守備スコア
df["Defense_Score"] = (
    df_scaled["STL"] * 0.3 +      # ボール奪取
    df_scaled["BLK"] * 0.3 +      # ブロック
    df_scaled["DREB"] * 0.3 +     # 守備リバウンド
    df_scaled["REB"] * 0.1        # 全体リバウンド補助
)

# クラスタリング
X = df[["Offense_Score", "Defense_Score"]].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# === クラスタラベル再定義 ===
cluster_labels, cluster_colors = {}, {}

for i in range(3):
    cdata = df[df["Cluster"] == i]
    avg_off = cdata["Offense_Score"].mean()
    avg_def = cdata["Defense_Score"].mean()

    if avg_off > avg_def and avg_off > 0:
        cluster_labels[i] = "Offense-Oriented"
        cluster_colors[i] = "red"
    elif avg_def > avg_off and avg_def > 0:
        cluster_labels[i] = "Defense-Oriented"
        cluster_colors[i] = "blue"
    else:
        cluster_labels[i] = "Balanced"
        cluster_colors[i] = "purple"

df["Playstyle"] = df["Cluster"].map(cluster_labels)

# 可視化
plt.figure(figsize=(12, 8))
for i in range(3):
    cdata = df[df["Cluster"] == i]
    plt.scatter(
        cdata["Offense_Score"], cdata["Defense_Score"],
        c=cluster_colors[i], label=cluster_labels[i],
        s=70, alpha=0.6
    )

# 八村塁ハイライト
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if not rui.empty:
    plt.scatter(
        rui["Offense_Score"], rui["Defense_Score"],
        s=250, color="gold", edgecolor="black", zorder=5, label="Rui Hachimura"
    )
    plt.text(
        rui["Offense_Score"].values[0] + 0.1,
        rui["Defense_Score"].values[0],
        "Rui Hachimura", color="black", fontsize=11, weight="bold"
    )

plt.title("NBA Player Playstyle Clustering (Offense vs Defense)", fontsize=15)
plt.xlabel("Offense Score (Standardized)")
plt.ylabel("Defense Score (Standardized)")
plt.legend()
plt.grid(alpha=0.3)

path = f"{plot_dir}/nba_playstyle_clustering_fixed.png"
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.close()

# 結果出力
print("クラスタ分析結果")
for i in range(3):
    cdata = df[df["Cluster"] == i]
    print(f"\n{cluster_labels[i]} ({len(cdata)}人):")
    print(f"  平均攻撃スコア: {cdata['Offense_Score'].mean():.3f}")
    print(f"  平均守備スコア: {cdata['Defense_Score'].mean():.3f}")
    print(f"  主な選手例: {cdata.nlargest(3, 'MIN')['PLAYER_NAME'].tolist()}")

if not rui.empty:
    print(f"\n八村塁の分類: {rui['Playstyle'].values[0]}")

print(f"\n出力完了: {path}")
