import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import os

# データ読み込み
input_path = "../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv"
df = pd.read_csv(input_path)
# 分析対象シーズン
target_seasons = ["2022-23", "2023-24", "2024-25"]
df = df[df["SEASON"].isin(target_seasons)].copy()

# 2P関連列を生成
df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)

# 説明変数を選択
features = [
    "MIN", "PTS",
    "FG2M", "FG2A", "FG2_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV"
]

X = df[features].fillna(0)
X_const = sm.add_constant(X)
# VIFの算出
vif_df = pd.DataFrame()
vif_df["Feature"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
# constを除外（定数項は不要）
vif_df = vif_df[vif_df["Feature"] != "const"].copy()
# 出力
print("=== 八村塁スタッツのVIF(多重共線性チェック)分析 ===")
print(vif_df.sort_values("VIF", ascending=False).to_string(index=False))
# 結果の解釈ガイド
print("\n判定ガイド:")
print("  VIF < 5    → 問題なし")
print("  5〜10     → 注意（相関強め）")
print("  >10       → 多重共線性あり")
# 出力ディレクトリ作成
output_dir = "../outputs/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. VIFの棒グラフ
plt.figure(figsize=(12, 8))
vif_sorted = vif_df.sort_values("VIF", ascending=True)
# 色分け（VIFの値に応じて）
colors = []
for vif_val in vif_sorted["VIF"]:
    if vif_val < 5:
        colors.append("green")
    elif vif_val < 10:
        colors.append("orange")
    else:
        colors.append("red")

bars = plt.barh(vif_sorted["Feature"], vif_sorted["VIF"], color=colors, alpha=0.7)
# VIF値をバーの上に表示
for i, (feature, vif_val) in enumerate(zip(vif_sorted["Feature"], vif_sorted["VIF"])):
    plt.text(vif_val + 0.1, i, f'{vif_val:.2f}', va='center', fontweight='bold')

# 基準線を追加
plt.axvline(x=5, color='orange', linestyle='--', alpha=0.8, label='VIF = 5 (Caution)')
plt.axvline(x=10, color='red', linestyle='--', alpha=0.8, label='VIF = 10 (Problem)')

plt.xlabel('VIF Value', fontsize=12)
plt.ylabel('Statistical Features', fontsize=12)
plt.title('Rui Hachimura Stats VIF (Multicollinearity Check) Analysis', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/rui_hachimura_vif_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. 相関ヒートマップ
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
mask = correlation_matrix.abs() < 0.6

sns.heatmap(correlation_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt='.2f',
            square=True,
            mask=mask,
            cbar_kws={"shrink": .8})

plt.title("Rui Hachimura Stats Correlation Matrix (|r| ≥ 0.6)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/rui_hachimura_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. 高VIF項目の詳細分析
high_vif = vif_df[vif_df["VIF"] > 5].sort_values("VIF", ascending=False)
if not high_vif.empty:
    print(f"\n【注意】VIF > 5 の項目:")
    for _, row in high_vif.iterrows():
        print(f"  {row['Feature']}: {row['VIF']:.2f}")
        
    # 高VIF項目間の相関を詳細表示
    high_vif_features = high_vif["Feature"].tolist()
    if len(high_vif_features) > 1:
        print(f"\n高VIF項目間の相関係数:")
        high_corr = X[high_vif_features].corr()
        print(high_corr.to_string())

print(f"\n出力完了:")
print(f"  VIF分析グラフ: {output_dir}/rui_hachimura_vif_analysis.png")
print(f"  相関ヒートマップ: {output_dir}/rui_hachimura_correlation_heatmap.png")
