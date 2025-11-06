import pandas as pd
import statsmodels.api as sm

# === データ読み込み ===
df = pd.read_csv("../outputs/csv/rui_hachimura_standard_stats_2019_2025.csv")

df["FG2A"] = df["FGA"] - df["FG3A"]
df["FG2M"] = df["FGM"] - df["FG3M"]
df["FG2_PCT"] = df.apply(lambda x: x["FG2M"] / x["FG2A"] if x["FG2A"] > 0 else 0, axis=1)

# 目的変数（得点）
y = df["PTS"]

# 説明変数候補
candidate_vars = ["MIN", "FG2A", "FG2_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT", "OREB", "TOV"]

# === ステップワイズ法（AIC基準） ===
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    """
    AIC基準の前進・後退ステップワイズ選択
    """
    included = list(initial_list)
    while True:
        changed = False
        # 追加ステップ
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(dtype=float)
        for new_var in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_var]])).fit()
            new_pval.loc[new_var] = model.pvalues[new_var]
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_var = new_pval.idxmin()
                included.append(best_var)
                changed = True
                if verbose:
                    print(f"追加: {best_var:>10s} (p={best_pval:.4f})")

        # 削除ステップ
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # const除外
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_var = pvalues.idxmax()
            included.remove(worst_var)
            if verbose:
                print(f"削除: {worst_var:>10s} (p={worst_pval:.4f})")

        if not changed:
            break

    return included

# 実行
selected_vars = stepwise_selection(df[candidate_vars], y)
print("\n=== 最終モデルに残った変数 ===")
print(selected_vars)

# 最終モデルを出力
final_model = sm.OLS(y, sm.add_constant(df[selected_vars])).fit()
print("\n=== 最終モデル ===")
print(final_model.summary())
