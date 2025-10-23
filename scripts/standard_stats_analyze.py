import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import os

# ファイル読み込み
input_path = "../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv"
standard_df = pd.read_csv(input_path)

# 対象シーズン抽出
target_seasons = ["2022-23", "2023-24", "2024-25"]
df = standard_df[standard_df["SEASON"].isin(target_seasons)].copy()

# 2P関連の算出
df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)

# 特徴量と目的変数（得点予測モデル）
scoring_features = [
    "MIN",
    "FG2A",
    "FG2_PCT",
    "FG3A",
    "FG3_PCT",
    "FTA",
    "FT_PCT",
    "OREB",
    "TOV"
]
X_scoring = df[scoring_features].fillna(0)
y_scoring = df["PTS"]

# インデックスをリセットして揃える
X_scoring = X_scoring.reset_index(drop=True)
y_scoring = y_scoring.reset_index(drop=True)

# 標準化
scaler_scoring = StandardScaler()
X_scoring_scaled = scaler_scoring.fit_transform(X_scoring)

# 定数項追加
X_scoring_scaled = sm.add_constant(X_scoring_scaled)

# 回帰モデル（p値つき）
model_scoring = sm.OLS(y_scoring, X_scoring_scaled).fit()

# 結果出力
print("得点予測モデル:")
print(model_scoring.summary())

# p値をデータフレーム化（有意差一覧）
summary_scoring_df = pd.DataFrame({
    "Feature": ["const"] + scoring_features,
    "Coefficient": model_scoring.params.values,
    "P_value": model_scoring.pvalues.values,
    "T_value": model_scoring.tvalues.values
}).sort_values("P_value")

print("\n得点予測モデル - 有意差結果")
print(summary_scoring_df)

# 出場時間予測モデル
minutes_features = [
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

X_minutes = df[minutes_features].fillna(0)
y_minutes = df["MIN"]

# インデックスをリセットして揃える
X_minutes = X_minutes.reset_index(drop=True)
y_minutes = y_minutes.reset_index(drop=True)

# 標準化
scaler_minutes = StandardScaler()
X_minutes_scaled = scaler_minutes.fit_transform(X_minutes)

# 定数項追加
X_minutes_scaled = sm.add_constant(X_minutes_scaled)

# 回帰モデル（p値つき）
model_minutes = sm.OLS(y_minutes, X_minutes_scaled).fit()

# 結果出力
print("出場時間予測モデル:")
print(model_minutes.summary())

# p値をデータフレーム化（有意差一覧）
summary_minutes_df = pd.DataFrame({
    "Feature": ["const"] + minutes_features,
    "Coefficient": model_minutes.params.values,
    "P_value": model_minutes.pvalues.values,
    "T_value": model_minutes.tvalues.values
}).sort_values("P_value")

print("\n出場時間予測モデル - 有意差結果")
print(summary_minutes_df)

# 有意水準0.05での帰無仮説検定結果
print("帰無仮説検定結果(α=0.05):")
print("得点予測モデル:")
for idx, row in summary_scoring_df.iterrows():
    if row["Feature"] != "const":
        if row["P_value"] < 0.05:
            print(f"  {row['Feature']}: p={row['P_value']:.4f} → 帰無仮説棄却（有意）")
        else:
            print(f"  {row['Feature']}: p={row['P_value']:.4f} → 帰無仮説採択（非有意）")

print("\n出場時間予測モデル:")
for idx, row in summary_minutes_df.iterrows():
    if row["Feature"] != "const":
        if row["P_value"] < 0.05:
            print(f"  {row['Feature']}: p={row['P_value']:.4f} → 帰無仮説棄却（有意）")
        else:
            print(f"  {row['Feature']}: p={row['P_value']:.4f} → 帰無仮説採択（非有意）")
