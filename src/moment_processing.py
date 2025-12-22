import os
import py7zr
import shutil
import json
import pandas as pd
import numpy as np
import shutil

GAME_LOGS_DIR = os.path.join('data', 'game_logs')
TEMP_LOGS_DIR = os.path.join('data', 'temp_logs')
MOMENT_DATA_DIR = os.path.join('data', 'moment_data')

def create_moment_df(fname_7z, save_to_csv=True):
    """
    Create a DataFrame of moments from the game provided by the given 7z file. The DF will not have
    duplicate moments (as determined by period/clock combination). 

    Args:
        fname_7z (str): File name (without directory information) of the 7z game moment file.
        save_to_csv (bool, optional): True if moment DF should be saved to CSV; False otherwise. Defaults to True.

    Returns:
        pd.DataFrame: Moments DataFrame.
    """
    full_7z_name = os.path.join(GAME_LOGS_DIR, fname_7z)
    with py7zr.SevenZipFile(full_7z_name, mode='r') as z:
        z.extractall(path=TEMP_LOGS_DIR)

    files = [f for f in os.listdir(TEMP_LOGS_DIR)]
    if len(files) == 0:
        print(f'{fname_7z} game has no data')
        return
    fpath = os.path.join(TEMP_LOGS_DIR, files[0])

    with open(fpath, 'r') as file:
        data = json.load(file)

    game_id = data['gameid']
    save_file = os.path.join(MOMENT_DATA_DIR, f'{game_id}.csv')
    # Do not repeat dataframe creation for games that already have been processed
    if os.path.exists(save_file):
        shutil.rmtree(TEMP_LOGS_DIR)
        return

    data_rows = []
    df_cols = [
        'event_id', 'moment_id', 'period', 'clock', 'shot_clock',
        'ball_x', 'ball_y', 'ball_z'
    ] + [
        col for i in range(10) for col in [f'player{i}_team_id', f'player{i}_id', f'player{i}_x', f'player{i}_y', f'player{i}_ball_dist']
    ] + [
        'closest_player_id1', 'closest_player_id2'
    ]

    for e in data['events']:
        for m_id, m in enumerate(e['moments']):
            locations = m[5]
            ball_x, ball_y, ball_z = locations[0][2:]
            player_list = [point for player_data in locations[1:] for point in player_data[:4]]
            if len(player_list) == 0:
                continue
            players = [
                (
                    player_list[i], player_list[i+1], player_list[i+2], player_list[i+3], np.sqrt((player_list[i+2] - ball_x)**2 + (player_list[i+3] - ball_y)**2)
                ) for i in range(0, len(player_list), 4)
            ]
            flattened_player_data = [d for player in players for d in player]
            closest_players = [player[0] for player in sorted(players, key=lambda p: p[3])[:2]]
            row = (e['eventId'], m_id, m[0], m[2], m[3], ball_x, ball_y, ball_z, *flattened_player_data, int(closest_players[0]), int(closest_players[1]))
            data_rows.append(row)
    
    shutil.rmtree(TEMP_LOGS_DIR)
    
    df = pd.DataFrame(data_rows, columns=df_cols)
    df_no_dup = df.drop_duplicates(subset=['period', 'clock'], keep='first')
    df_no_dup_sort = df_no_dup.sort_values(by=['period', 'clock'], ascending=[True, False])

    if save_to_csv:
        os.makedirs(MOMENT_DATA_DIR, exist_ok=True)
        save_file = os.path.join(MOMENT_DATA_DIR, f'{game_id}.csv')
        df_no_dup_sort.to_csv(save_file, index=False)
    
    return df_no_dup_sort

def moment_df_driver(fname=None, save=True):
    """
    Driver function to create moment DataFrame CSV files.

    Args:
        fname (str, optional): Game 7zip file name for which DataFrame should be created. 
         If None, then DataFrame is created for all valid stored game files. Defaults to None.
        save (bool, optional): True if moment DataFrames should be stored; False otherwise. Defaults to True.
    """
    if fname is not None:
        return create_moment_df(fname, save)
    
    proc_count = 0

    for root, dirs, files in os.walk(GAME_LOGS_DIR):
        for file in files:
            create_moment_df(fname_7z=file, save_to_csv=save)
            proc_count += 1

            if proc_count % 25 == 0:
                print(f'Created moment DataFrame for {proc_count} games')


if __name__=='__main__':
    moment_df_driver()