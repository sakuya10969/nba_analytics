import pandas as pd

# データ読み込み
df = pd.read_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv")
# 八村塁を抽出
rui = df[df["PLAYER_NAME"].str.contains("Hachimura", case=False, na=False)]
if rui.empty:
    raise ValueError("八村塁のデータが見つかりません。")
# 対象列
metrics = ["FG_PCT", "FG3_PCT"]

for metric in metrics:
    if metric not in df.columns:
        print(f"{metric} がデータに含まれていません。")
        continue
    # 八村の値
    rui_value = rui.iloc[0][metric]
    # パーセンタイル計算
    percentile = (df[metric] < rui_value).mean() * 100
    rank = df[metric].rank(pct=True).loc[rui.index[0]] * 100

    print(f"{metric}: 八村塁 = {rui_value:.3f}")
    print(f"  → 全選手中 上位 {100 - percentile:.2f}%（下位 {percentile:.2f}%)")
