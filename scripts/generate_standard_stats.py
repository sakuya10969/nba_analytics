from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import os

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
    "SEASON", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS",  # シーズン、試合日、対戦カード、勝敗、出場時間、得点
    "FGM", "FGA", "FG_PCT",  # フィールドゴール成功数、試投数、成功率
    "FG3M", "FG3A", "FG3_PCT",  # 3ポイント成功数、試投数、成功率
    "FTM", "FTA", "FT_PCT",  # フリースロー成功数、試投数、成功率
    "OREB", "DREB", "REB",  # オフェンスリバウンド、ディフェンスリバウンド、総リバウンド
    "AST", "STL", "BLK", "TOV", "PF",  # アシスト、スティール、ブロック、ターンオーバー、ファウル
    "PLUS_MINUS"  # プラスマイナス
]]
# 出力ディレクトリ作成
output_dir = "../outputs/csv"
os.makedirs(output_dir, exist_ok=True)
# CSVに保存
games_df.to_csv(f"{output_dir}/rui_hachimura_standard_stats_2019_2025.csv", index=False)

print("取得データ件数:", len(games_df))
print(games_df.head())