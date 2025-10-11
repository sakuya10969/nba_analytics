from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

# データ取得
print(f"Fetching player stats for 2024-25...")
data = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2024-25",
    per_mode_detailed="PerGame"
)
df = data.get_data_frames()[0]

# フィルタ: 平均出場時間20分以上
df_filtered = df[df["MIN"] >= 20].reset_index(drop=True)

# 利用可能なカラムを確認
print("Available columns:", df_filtered.columns.tolist())

# 必要なカラムを抽出（存在するもののみ）
columns = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN",
    "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "AST", "TOV", "REB", "PLUS_MINUS"
]

# 存在するカラムのみを選択
available_columns = [col for col in columns if col in df_filtered.columns]
print(f"Using columns: {available_columns}")

df_selected = df_filtered[available_columns].copy()

# 型変換とNaN処理
df_selected = df_selected.fillna(0)

# 統計情報を確認
print(f"Total players (>=20 min): {len(df_selected)}")

# 保存
df_selected.to_csv("../outputs/csv/nba_players_2024_2025_20min_clustering.csv", index=False)
print(f"Saved to nba_players_2024_2025_20min_clustering.csv")
