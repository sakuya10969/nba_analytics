from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time

# 2024-25シーズンのみ
season = "2024-25"

print(f"Fetching season {season} ...")

data = leaguedashplayerstats.LeagueDashPlayerStats(
    season=season,
    season_type_all_star="Regular Season",
    measure_type_detailed_defense="Advanced",
    per_mode_detailed="PerGame"
)

df = data.get_data_frames()[0]
df["SEASON"] = season

# GPが30超えのプレイヤーにフィルタリング
df_filtered = df[df["GP"] >= 30].reset_index(drop=True)

# 必要なカラムを抽出
columns = [
    "SEASON",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "OFF_RATING",
    "DEF_RATING", 
    "NET_RATING",
    "AST_PCT",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE"
]

# 存在するカラムのみを選択
available_columns = [col for col in columns if col in df_filtered.columns]
df_selected = df_filtered[available_columns].copy()

print(f"Total players (GP >= 30): {len(df_selected)}")

# CSV出力
df_selected.to_csv("../outputs/csv/nba_players_advanced_2024_2025_30games.csv", index=False)
print("Saved to nba_players_advanced_2024_2025_30games.csv")
