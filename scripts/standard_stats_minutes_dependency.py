from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# データ読み込み
standard_df = pd.read_csv("../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv")
# 出力ディレクトリ
output_dir = "../outputs/plots"
os.makedirs(output_dir, exist_ok=True)
# 2022–2025シーズンのみ抽出
target_seasons = ["2022-23", "2023-24", "2024-25"]
df = standard_df[standard_df["SEASON"].isin(target_seasons)].copy()
# 2P試投数と成功率を作成
df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)
# 特徴量（出場時間に影響する要因）
features = [
    "PTS",
    "FG2M",
    "FG2_PCT",
    "FG3M",
    "FG3_PCT",
    "FTM",
    "FT_PCT",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
]

X = df[features].fillna(0)
y = df["MIN"]
# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 線形回帰モデル
model = LinearRegression()
model.fit(X_scaled, y)
# 係数
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
coef_df["Abs"] = np.abs(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Abs", ascending=False)

print(coef_df)
print(f"R²: {model.score(X_scaled, y):.3f}")
# 可視化
plt.figure(figsize=(8, 6))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="blue")
plt.xlabel("Coefficient")
plt.title("Rui Hachimura - Minutes Dependency Model (2022–2025)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/rui_hachimura_minutes_dependency_2022_2025.png")
plt.close()

print(f"Saved to {output_dir}/rui_hachimura_minutes_dependency_2022_2025.png")
