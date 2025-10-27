import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

# データ読み込み
standard_df = pd.read_csv("../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv")
output_dir = "../outputs/plots"
os.makedirs(output_dir, exist_ok=True)
# 対象シーズン
target_seasons = ["2022-23", "2023-24", "2024-25"]
df = standard_df[standard_df["SEASON"].isin(target_seasons)].copy()
# 2P関連列を生成
df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)
# 特徴量と目的変数
features = ["FG2A", "FG2_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT", "OREB"]
X = df[features].fillna(0)
y = df["PTS"]
# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 線形回帰モデル
model = LinearRegression()
model.fit(X_scaled, y)
# 係数データフレーム
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
coef_df["Abs"] = np.abs(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Abs", ascending=True)  # 可視化は下→上のため昇順
# モデルスコア
r2 = model.score(X_scaled, y)
print(coef_df)
print(f"R²: {r2:.3f}")
# 可視化設定
plt.figure(figsize=(9, 6))
# 色分け（正: 赤 / 負: 青）
colors = ["red" if c > 0 else "blue" for c in coef_df["Coefficient"]]

bars = plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black")
# 値ラベルを右横に描画
for bar, value in zip(bars, coef_df["Coefficient"]):
    plt.text(
        bar.get_width() + (0.05 if value > 0 else -0.05),
        bar.get_y() + bar.get_height()/2,
        f"{value:.2f}",
        va="center",
        ha="left" if value > 0 else "right",
        fontsize=10,
        color="black"
    )

# 軸・タイトル整備
plt.axvline(0, color="black", linewidth=1)
plt.xlim(-1, 4)
plt.title("Rui Hachimura - Scoring Dependency Model (2022–2025)", fontsize=13, fontweight="bold")
plt.xlabel("Coefficient", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.text(
    0.95, 0.05,
    f"$R^2$ = {r2:.3f}",
    ha="right", va="bottom",
    transform=plt.gca().transAxes,
    fontsize=11,
    color="black"
)
plt.tight_layout()
# 保存
output_path = f"{output_dir}/rui_hachimura_scoring_dependency_2022_2025.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved enhanced plot to {output_path}")
