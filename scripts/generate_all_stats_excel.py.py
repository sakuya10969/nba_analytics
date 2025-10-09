from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd

# 八村塁のplayer_idを取得
player_dict = players.find_players_by_full_name("Rui Hachimura")
rui_id = player_dict[0]['id']   # 八村のIDを取得

# 2019-20から2024-25シーズンのログをまとめて取得
seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24","2024-25"]

all_games = []
for season in seasons:
    gamelog = playergamelog.PlayerGameLog(player_id=rui_id, season=season)
    df = gamelog.get_data_frames()[0]
    df["SEASON"] = season
    all_games.append(df)

# 結合
games_df = pd.concat(all_games, ignore_index=True)

# 必要な列だけ抽出（例：日付、対戦チーム、得点、リバウンド、アシスト、出場時間など）
games_df = games_df[[
    "SEASON", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF",
    "PLUS_MINUS"
]]
# Excelに保存
games_df.to_excel("../outputs/rui_hachimura_games_all_stats_2019_2025.xlsx", index=False)

print("取得データ件数:", len(games_df))
print(games_df.head())