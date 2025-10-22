from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time

# データ取得
print(f"Fetching player stats for 2024-25...")

try:
    data = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2024-25",
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Traditional",
        per_mode_detailed="PerGame"
    )
    df = data.get_data_frames()[0]
except Exception as e:
    print(f"Error fetching data: {e}")
    print("Retrying with different parameters...")
    time.sleep(2)
    
    try:
        data = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2024-25",
            season_type_all_star="Regular Season"
        )
        df = data.get_data_frames()[0]
    except Exception as e2:
        print(f"Second attempt failed: {e2}")
        print("Trying with minimal parameters...")
        time.sleep(2)
        
        data = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2024-25"
        )
        df = data.get_data_frames()[0]

# フィルタ: 出場試合数30試合以上
df_filtered = df[df["GP"] >= 30].reset_index(drop=True)

# 利用可能なカラムを確認
print("Available columns:", df_filtered.columns.tolist())

# 必要なカラムを抽出
columns = [
    "PLAYER_ID",        # プレイヤーID
    "PLAYER_NAME",      # プレイヤー名
    "TEAM_ABBREVIATION", # チーム略称
    "AGE",              # 年齢
    "GP",               # 出場試合数 (Games Played)
    "MIN",              # 平均出場時間 (Minutes)
    "PTS",              # 平均得点 (Points)
    "FGM",              # フィールドゴール成功数 (Field Goals Made)
    "FGA",              # フィールドゴール試投数 (Field Goals Attempted)
    "FG_PCT",           # フィールドゴール成功率 (Field Goal Percentage)
    "FG3M",             # 3ポイント成功数 (3-Point Field Goals Made)
    "FG3A",             # 3ポイント試投数 (3-Point Field Goals Attempted)
    "FG3_PCT",          # 3ポイント成功率 (3-Point Field Goal Percentage)
    "FTM",              # フリースロー成功数 (Free Throws Made)
    "FTA",              # フリースロー試投数 (Free Throws Attempted)
    "FT_PCT",           # フリースロー成功率 (Free Throw Percentage)
    "OREB",             # オフェンシブリバウンド数 (Offensive Rebounds)
    "DREB",             # ディフェンシブリバウンド数 (Defensive Rebounds)
    "REB",              # リバウンド数 (Rebounds)
    "AST",              # アシスト数 (Assists)
    "STL",              # スティール数 (Steals)
    "BLK",              # ブロック数 (Blocks)
    "TOV",              # ターンオーバー数 (Turnovers)
    "PF",               # パーソナルファウル数 (Personal Fouls)
    "PLUS_MINUS"        # プラスマイナス (Plus/Minus)
]

# 存在するカラムのみを選択
available_columns = [col for col in columns if col in df_filtered.columns]
print(f"Using columns: {available_columns}")

df_selected = df_filtered[available_columns].copy()

# 型変換とNaN処理
df_selected = df_selected.fillna(0)

# 統計情報を確認
print(f"Total players (>=30 games): {len(df_selected)}")

# 保存
df_selected.to_csv("../outputs/csv/nba_players_2024_2025_30games_clustering.csv", index=False)
print(f"Saved to nba_players_2024_2025_30games_clustering.csv")
