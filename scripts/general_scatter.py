import os
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
plot_dir = "../outputs/plots"
os.makedirs(plot_dir, exist_ok=True)
# 3ポイント関連の特徴量のみを使用
features_3pt = ["FG3A", "FG3_PCT"]
# 欠損値を0で埋める
df_clean_3pt = df[features_3pt].fillna(0)
# 散布図の可視化（3ポイント）
plt.figure(figsize=(12, 8))
# 全選手をプロット
plt.scatter(
    df_clean_3pt["FG3A"], df_clean_3pt["FG3_PCT"],
    c="green", s=50, alpha=0.6, edgecolor="gray", linewidth=0.5
)
# 八村塁ハイライト
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if not rui.empty:
    plt.scatter(
        rui["FG3A"], rui["FG3_PCT"],
        s=250, color="blue", edgecolor="black", zorder=5, label="Rui Hachimura"
    )

plt.title("NBA Player 3-Point Shooting", fontsize=15)
plt.xlabel("3-Point Attempts(FG3A)")
plt.ylabel("3-Point Percentage (FG3_PCT)")
plt.legend()
plt.grid(alpha=0.3)

path_3pt = f"{plot_dir}/nba_3pt_scatter.png"
plt.savefig(path_3pt, bbox_inches="tight", dpi=300)
plt.close()

# フィールドゴール関連の特徴量を使用
features_fg = ["FGA", "FG_PCT"]
# 欠損値を0で埋める
df_clean_fg = df[features_fg].fillna(0)
# 散布図の可視化（フィールドゴール
plt.figure(figsize=(12, 8))
# 全選手をプロット
plt.scatter(
    df_clean_fg["FGA"], df_clean_fg["FG_PCT"],
    c="green", s=50, alpha=0.6, edgecolor="gray", linewidth=0.5
)
# 八村塁ハイライト
if not rui.empty:
    plt.scatter(
        rui["FGA"], rui["FG_PCT"],
        s=250, color="blue", edgecolor="black", zorder=5, label="Rui Hachimura"
    )

plt.title("NBA Player Field Goal Shooting", fontsize=15)
plt.xlabel("Field Goal Attempts (FGA)")
plt.ylabel("Field Goal Percentage (FG_PCT)")
plt.legend()
plt.grid(alpha=0.3)

path_fg = f"{plot_dir}/nba_fg_scatter.png"
plt.savefig(path_fg, bbox_inches="tight", dpi=300)
plt.close()
# 結果出力
print("3ポイントシュート散布図分析結果")
print(f"総選手数: {len(df)}人")
print(f"平均3P試行数: {df_clean_3pt['FG3A'].mean():.3f}")
print(f"平均3P成功率: {df_clean_3pt['FG3_PCT'].mean():.3f}")

print("\nフィールドゴール散布図分析結果")
print(f"平均FG試行数: {df_clean_fg['FGA'].mean():.3f}")
print(f"平均FG成功率: {df_clean_fg['FG_PCT'].mean():.3f}")

if not rui.empty:
    print(f"\n八村塁の3P試行数: {rui['FG3A'].values[0]:.1f}")
    print(f"八村塁の3P成功率: {rui['FG3_PCT'].values[0]:.3f}")
    print(f"八村塁のFG試行数: {rui['FGA'].values[0]:.1f}")
    print(f"八村塁のFG成功率: {rui['FG_PCT'].values[0]:.3f}")

print(f"\n出力完了:")
print(f"  3ポイント散布図: {path_3pt}")
print(f"  フィールドゴール散布図: {path_fg}")
