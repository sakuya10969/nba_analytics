import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 出力先ディレクトリ
output_dir = "../outputs/plots"
os.makedirs(output_dir, exist_ok=True)

input_path = "../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv"
df = pd.read_csv(input_path)
target_seasons = ["2022-23", "2023-24", "2024-25"]
df = df[df["SEASON"].isin(target_seasons)].copy()

df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)

def run_regression(X, y, label):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()

    summary_df = pd.DataFrame({
        "Feature": ["const"] + list(X.columns),
        "Coefficient": model.params.values,
        "P_value": model.pvalues.values,
        "T_value": model.tvalues.values
    }).sort_values("P_value")

    print(f"\n=== {label} ===")
    print(f"帰無仮説: 各変数の係数は0である")
    print(f"対立仮説: 各変数の係数は0ではない")
    print(f"有意水準: α = 0.05")
    
    # 帰無仮説棄却の判定
    for _, row in summary_df.iterrows():
        if row["Feature"] != "const":
            if row["P_value"] < 0.05:
                print(f"{row['Feature']}: p={row['P_value']:.4f} < 0.05 → 帰無仮説棄却 (有意)")
            else:
                print(f"{row['Feature']}: p={row['P_value']:.4f} ≥ 0.05 → 帰無仮説採択 (非有意)")

    return model, summary_df

def plot_coef_heatmap(summary_df, title, save_path):
    plt.figure(figsize=(8, len(summary_df)*0.5 + 1))
    # p値に基づいて色分けのためのマスクを作成
    data_for_heatmap = summary_df.set_index("Feature")[["Coefficient", "P_value"]]
    # 有意性に基づいて色を変える（p < 0.05で濃い色、p >= 0.05で薄い色）
    # p値を0-1の範囲で正規化し、有意性に基づいて調整
    p_values = data_for_heatmap["P_value"].values
    significance_mask = p_values < 0.05
    # 係数とp値を別々にプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, len(summary_df)*0.5 + 1))
    # 係数のヒートマップ（有意性に基づいて透明度を調整）
    coef_data = data_for_heatmap[["Coefficient"]].copy()
    sns.heatmap(
        coef_data,
        annot=True, 
        cmap="RdBu_r", 
        fmt=".3f", 
        cbar=True,
        ax=ax1,
        center=0
    )
    ax1.set_title("Coefficient", fontsize=12, weight="bold")
    # p値のヒートマップ（0.05を境界として色分け）
    p_data = data_for_heatmap[["P_value"]].copy()
    # p値を色分けのために変換（0.05未満は1、以上は0.3として薄く表示）
    p_colors = np.where(p_values.reshape(-1, 1) < 0.05, 1.0, 0.3)
    
    sns.heatmap(
        p_data,
        annot=True, 
        cmap="Reds", 
        fmt=".4f", 
        cbar=True,
        ax=ax2,
        vmin=0,
        vmax=1
    )
    ax2.set_title("P-value (Dark: p<0.05, Light: p≥0.05)", fontsize=12, weight="bold")
    # 有意性を視覚的に強調するため、p<0.05の行に枠線を追加
    for i, (idx, row) in enumerate(data_for_heatmap.iterrows()):
        if row["P_value"] < 0.05:
            # 係数のヒートマップに枠線
            ax1.add_patch(plt.Rectangle((0, i), 1, 1, fill=False, edgecolor='black', lw=3))
            # p値のヒートマップに枠線
            ax2.add_patch(plt.Rectangle((0, i), 1, 1, fill=False, edgecolor='black', lw=3))
    
    plt.suptitle(f"{title}", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()

# 得点モデル
scoring_features = ["MIN", "FG2A", "FG2_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT", "OREB", "TOV"]
X_scoring = df[scoring_features].fillna(0)
y_scoring = df["PTS"]
model_scoring, summary_scoring_df = run_regression(X_scoring, y_scoring, "scoring model")

plot_coef_heatmap(summary_scoring_df, "scoring model", os.path.join(output_dir, "scoring_coef_heatmap.png"))

# 出場時間モデル
minutes_features = ["PTS", "FG2M", "FG2_PCT", "FG3M", "FG3_PCT", "FTM", "FT_PCT", "OREB", "DREB", "AST", "STL", "BLK", "TOV"]
X_minutes = df[minutes_features].fillna(0)
y_minutes = df["MIN"]
model_minutes, summary_minutes_df = run_regression(X_minutes, y_minutes, "minutes model")

plot_coef_heatmap(summary_minutes_df, "minutes model", os.path.join(output_dir, "minutes_coef_heatmap.png"))
