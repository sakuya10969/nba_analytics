from nba_api.stats.endpoints import playergamelog, boxscoreadvancedv2
from nba_api.stats.static import players
import pandas as pd
import time

# === 八村塁のplayer_idを取得 ===
player_dict = players.find_players_by_full_name("Rui Hachimura")
rui_id = player_dict[0]['id']
print(f"Rui Hachimura PLAYER_ID: {rui_id}")

# === 対象シーズン ===
seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

all_games = []

for season in seasons:
    print(f"Fetching GAME LOG for {season}...")
    gamelog = playergamelog.PlayerGameLog(player_id=rui_id, season=season)
    df_log = gamelog.get_data_frames()[0]
    df_log["SEASON"] = season
    all_games.append(df_log)

# === 全シーズン結合 ===
games_df = pd.concat(all_games, ignore_index=True)

# === アドバンスドスタッツ格納用 ===
advanced_stats_list = []

# === 各試合のGAME_IDを使って詳細データ取得 ===
for i, row in games_df.iterrows():
    game_id = row["Game_ID"]
    try:
        box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        df_adv = box.get_data_frames()[0]

        # 八村の行だけ抽出
        player_row = df_adv[df_adv["PLAYER_ID"] == rui_id]

        if not player_row.empty:
            advanced_stats_list.append({
                "SEASON": row["SEASON"],
                "GAME_ID": game_id,
                "GAME_DATE": row["GAME_DATE"],
                "MATCHUP": row["MATCHUP"],
                "MIN": player_row["MIN"].values[0],
                "OFF_RATING": player_row["OFF_RATING"].values[0],
                "DEF_RATING": player_row["DEF_RATING"].values[0],
                "NET_RATING": player_row["NET_RATING"].values[0],
                "AST_PCT": player_row["AST_PCT"].values[0],
                "AST_RATIO": player_row["AST_RATIO"].values[0],
                "OREB_PCT": player_row["OREB_PCT"].values[0],
                "DREB_PCT": player_row["DREB_PCT"].values[0],
                "REB_PCT": player_row["REB_PCT"].values[0],
                "TO_RATIO": player_row["TM_TOV_PCT"].values[0],
                "EFG_PCT": player_row["EFG_PCT"].values[0],
                "TS_PCT": player_row["TS_PCT"].values[0],
                "USG_PCT": player_row["USG_PCT"].values[0],
                "PACE": player_row["PACE"].values[0],
                "PIE": player_row["PIE"].values[0],
            })
        else:
            print(f"No data for GAME_ID {game_id}")
    except Exception as e:
        print(f"Error fetching GAME_ID {game_id}: {e}")
        time.sleep(3)
        continue

    # API制限対策で少しウェイト（重要）
    time.sleep(1.2)

# === DataFrame化 ===
adv_df = pd.DataFrame(advanced_stats_list)

# === CSV出力 ===
adv_df.to_excel("../outputs/excel/rui_hachimura_advanced_stats_2019_2025.xlsx", index=False)

print(f"\n✅ 取得完了: {len(adv_df)} 試合分のアドバンスドスタッツを保存しました。")
print(adv_df.head())
