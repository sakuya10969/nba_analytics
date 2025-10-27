import matplotlib.pyplot as plt
import pandas as pd
import json
from nba_api.stats.endpoints import shotchartdetail
import numpy as np
import os
import sys
sys.path.append('../')
from utils.drawcount import draw_court

# 八村塁のプレイヤーID
player_id = '1629060'  # Rui Hachimura
# 2つの期間に分けてショットチャートを作成
periods = {
    'early': ['2019-20', '2020-21', '2021-22'],
    'recent': ['2022-23', '2023-24', '2024-25']
}

for period_name, seasons in periods.items():
    print(f"\n=== {period_name.upper()} PERIOD ({seasons[0]} to {seasons[-1]}) ===")
    # 全シーズンのデータを格納するリスト
    all_shot_data = []
    # 各シーズンのショットチャートデータを取得
    for season in seasons:
        print(f"取得中: {season}シーズン")
        try:
            shot_chart = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=season,
                season_type_all_star='Regular Season',
                context_measure_simple='FGA'
            )
            
            shot_chart_df = shot_chart.get_data_frames()[0]
            shot_chart_df['SEASON'] = season  # シーズン情報を追加
            all_shot_data.append(shot_chart_df)
        except Exception as e:
            print(f"エラー: {season}シーズンのデータ取得に失敗 - {e}")
            continue
    # データが取得できた場合のみ処理を続行
    if not all_shot_data:
        print(f"{period_name}期間のデータが取得できませんでした。")
        continue
    # 全データを結合
    combined_shot_df = pd.concat(all_shot_data, ignore_index=True)
    # ショット成功/失敗を分ける
    made = combined_shot_df[combined_shot_df.SHOT_MADE_FLAG == 1]
    miss = combined_shot_df[combined_shot_df.SHOT_MADE_FLAG == 0]
    # 2P/3P成功率を計算（全シーズン合計）
    df_2pfg = combined_shot_df[combined_shot_df.SHOT_TYPE == "2PT Field Goal"]
    df_2pfg_made = df_2pfg[df_2pfg.SHOT_MADE_FLAG == 1]
    df_3pfg = combined_shot_df[combined_shot_df.SHOT_TYPE == "3PT Field Goal"]
    df_3pfg_made = df_3pfg[df_3pfg.SHOT_MADE_FLAG == 1]

    print(f"\n全シーズン合計統計:")
    if len(df_2pfg) > 0:
        print("2PFG%     " + str(round(len(df_2pfg_made)/len(df_2pfg)*100, 3)) + "%")
    if len(df_3pfg) > 0:
        print("3PFG%     " + str(round(len(df_3pfg_made)/len(df_3pfg)*100, 3)) + "%")

    # シーズン別統計も表示
    for season in seasons:
        season_data = combined_shot_df[combined_shot_df.SEASON == season]
        if len(season_data) > 0:
            season_2pfg = season_data[season_data.SHOT_TYPE == "2PT Field Goal"]
            season_2pfg_made = season_2pfg[season_2pfg.SHOT_MADE_FLAG == 1]
            season_3pfg = season_data[season_data.SHOT_TYPE == "3PT Field Goal"]
            season_3pfg_made = season_3pfg[season_3pfg.SHOT_MADE_FLAG == 1]
            
            print(f"\n{season}シーズン:")
            if len(season_2pfg) > 0:
                print("2PFG%     " + str(round(len(season_2pfg_made)/len(season_2pfg)*100, 3)) + "%")
            if len(season_3pfg) > 0:
                print("3PFG%     " + str(round(len(season_3pfg_made)/len(season_3pfg)*100, 3)) + "%")

    # プロット作成
    plt.figure(figsize=(12, 11))
    plt.scatter(made.LOC_X, made.LOC_Y, c="green", alpha=0.7, s=50, label='Made')
    plt.scatter(miss.LOC_X, miss.LOC_Y, c="red", alpha=0.7, s=50, label='Miss')
    # コートを描画
    draw_court(outer_lines=True)
    # 軸の設定
    plt.xlim(-250, 250)
    plt.ylim(422.5, -47.5)
    # タイトルを期間に応じて設定
    title_text = f'Rui Hachimura Shot Chart - {seasons[0]} to {seasons[-1]}'
    plt.title(title_text, fontsize=16, weight='bold')
    plt.legend()
    # 出力ディレクトリ作成
    plot_dir = "../outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)
    # ファイル名を期間に応じて設定
    filename = f"rui_hachimura_shot_chart_{seasons[0].replace('-', '_')}_{seasons[-1].replace('-', '_')}.png"
    # 保存
    plt.savefig(f"{plot_dir}/{filename}", 
                bbox_inches="tight", dpi=300)
    print(f"\n出力完了: {plot_dir}/{filename}")
    
    plt.close()  # メモリ節約のためプロットを閉じる
