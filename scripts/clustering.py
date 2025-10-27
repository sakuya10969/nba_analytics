import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)
# 特徴量（得点型 vs アシスト型に必要なスタッツのみ）
scoring_features = ["PTS", "FGM", "FG_PCT", "FG3M", "FG3_PCT", "FTM", "FT_PCT"]
playmaking_features = ["AST", "TOV"]
# 標準化
df_scaled = df.copy()

scoring_scaler = StandardScaler()
playmaking_scaler = StandardScaler()

df_scaled[scoring_features] = scoring_scaler.fit_transform(df[scoring_features].fillna(0))
df_scaled[playmaking_features] = playmaking_scaler.fit_transform(df[playmaking_features].fillna(0))
# 得点スコア
df["Scoring_Score"] = (
    df_scaled["PTS"] * 0.4 +       # 得点量
    df_scaled["FGM"] * 0.2 +       # シュート成功数
    df_scaled["FG_PCT"] * 0.15 +   # シュート効率
    df_scaled["FG3M"] * 0.1 +      # 3ポイント成功数
    df_scaled["FG3_PCT"] * 0.05 +  # 3ポイント効率
    df_scaled["FTM"] * 0.05 +      # フリースロー成功数
    df_scaled["FT_PCT"] * 0.05     # フリースロー効率
)
# アシストスコア
df["Playmaking_Score"] = (
    df_scaled["AST"] * 0.8 -       # アシスト数
    df_scaled["TOV"] * 0.2         # ターンオーバー（マイナス要因）
)
# クラスタリング
X = df[["Scoring_Score", "Playmaking_Score"]].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
# === クラスタラベル再定義 ===
cluster_labels, cluster_colors = {}, {}

for i in range(3):
    cdata = df[df["Cluster"] == i]
    avg_scoring = cdata["Scoring_Score"].mean()
    avg_playmaking = cdata["Playmaking_Score"].mean()

    if avg_scoring > avg_playmaking and avg_scoring > 0:
        cluster_labels[i] = "Scorer"
        cluster_colors[i] = "red"
    elif avg_playmaking > avg_scoring and avg_playmaking > 0:
        cluster_labels[i] = "Playmaker"
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
        cdata["Scoring_Score"], cdata["Playmaking_Score"],
        c=cluster_colors[i], label=cluster_labels[i],
        s=70, alpha=0.6
    )
# 八村塁ハイライト
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if not rui.empty:
    plt.scatter(
        rui["Scoring_Score"], rui["Playmaking_Score"],
        s=250, color="gold", edgecolor="black", zorder=5, label="Rui Hachimura"
    )
    plt.text(
        rui["Scoring_Score"].values[0] + 0.1,
        rui["Playmaking_Score"].values[0],
        "Rui Hachimura", color="black", fontsize=11, weight="bold"
    )

plt.title("NBA Player Playstyle Clustering (Scorer vs Playmaker)", fontsize=15)
plt.xlabel("Scoring Score")
plt.ylabel("Playmaking Score")
plt.legend()
plt.grid(alpha=0.3)

path = f"{plot_dir}/nba_playstyle_clustering.png"
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.close()
# 結果出力
print("クラスタ分析結果")
for i in range(3):
    cdata = df[df["Cluster"] == i]
    print(f"\n{cluster_labels[i]} ({len(cdata)}人):")
    print(f"  平均得点スコア: {cdata['Scoring_Score'].mean():.3f}")
    print(f"  平均アシストスコア: {cdata['Playmaking_Score'].mean():.3f}")
    print(f"  主な選手例: {cdata.nlargest(3, 'MIN')['PLAYER_NAME'].tolist()}")

if not rui.empty:
    print(f"\n八村塁の分類: {rui['Playstyle'].values[0]}")

print(f"\n出力完了: {path}")
