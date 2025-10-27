import os
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)

# 3ポイント関連の特徴量のみを使用
features = ["FG3A", "FG3_PCT"]

# 欠損値を0で埋める
df_clean = df[features].fillna(0)

# 散布図の可視化
plt.figure(figsize=(12, 8))

# 全選手をプロット
plt.scatter(
    df_clean["FG3A"], df_clean["FG3_PCT"],
    c="red", s=50, alpha=0.6, edgecolor="gray", linewidth=0.5
)

# 八村塁ハイライト
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if not rui.empty:
    plt.scatter(
        rui["FG3A"], rui["FG3_PCT"],
        s=150, color="blue", edgecolor="black", zorder=5, label="Rui Hachimura"
    )
    plt.text(
        rui["FG3A"].values[0] + 15, 
        rui["FG3_PCT"].values[0], 
        "Rui Hachimura", 
        color="black", 
        fontsize=11, 
        weight="bold",
        va="center"
    )

plt.title("NBA Player 3-Point Shooting", fontsize=15)
plt.xlabel("3-Point Attempts(FG3A)")
plt.ylabel("3-Point Percentage (FG3_PCT)")
plt.legend()
plt.grid(alpha=0.3)

path = f"{plot_dir}/nba_3pt_scatter.png"
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.close()

# 結果出力
print("3ポイントシュート散布図分析結果")
print(f"総選手数: {len(df)}人")
print(f"平均3P試行数: {df_clean['FG3A'].mean():.3f}")
print(f"平均3P成功率: {df_clean['FG3_PCT'].mean():.3f}")

if not rui.empty:
    print(f"\n八村塁の3P試行数: {rui['FG3A'].values[0]:.1f}")
    print(f"八村塁の3P成功率: {rui['FG3_PCT'].values[0]:.3f}")

print(f"\n出力完了: {path}")
