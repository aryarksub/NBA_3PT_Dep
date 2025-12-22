import pandas as pd
import os

from moment_processing import MOMENT_DATA_DIR

# PBP data source: https://www.kaggle.com/datasets/wyattowalsh/basketball?resource=download
FULL_PBP_FILE = os.path.join('data', 'play_by_play.csv')
GAME_INFO_FILE = os.path.join('data', 'game_info.csv')
BASIC_SHOTS_FILE = os.path.join('data', 'pbp_shots_basic.csv')
SHOT_HISTORY_FILE = os.path.join('data', 'pbp_shots_history.csv')

def get_stored_moment_game_ids():
    """
    Get the game IDs for all games with moment data that have been stored

    Returns:
        list: List of stored game ID strings
    """
    return [file.split('.')[0][2:] for root, dirs, files in os.walk(MOMENT_DATA_DIR) for file in files]

def create_pbp_range_dataset(start_date, end_date, exclude_non_stored=True, save_file=None):
    """
    Create a filtered play-by-play dataset that only considers shot-related plays in 
    games played on days that fall in between the given start/end date range. Optionally
    ignore games that do not have stored moment data.

    Args:
        start_date (str): Start date ('YYYY-MM-DD') of range.
        end_date (str): End date ('YYYY-MM-DD') of range.
        exclude_non_stored (bool, optional): True if games without moment data should
         be ignored; False otherwise. Defaults to True.

    Returns:
        pd.DataFrame: Filtered shots play-by-play DataFrame.
    """
    pbp_df = pd.read_csv(FULL_PBP_FILE)

    # if exclude_non_stored, keep only games that have moment DFs stored
    if exclude_non_stored:
        # get list of all stored game IDs for moment CSV files
        game_ids = get_stored_moment_game_ids()
        pbp_df = pbp_df[pbp_df['game_id'].astype(str).isin(game_ids)]

    # filter for only shot-related plays (made/missed shots)
    pbp_shots_df = pbp_df[pbp_df['eventmsgtype'].isin([1,2])]
    
    games_df = pd.read_csv(GAME_INFO_FILE)
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # filter games to only range
    filtered_games = games_df[(games_df['game_date'] >= start_date) & (games_df['game_date'] <= end_date)]

    # join pbp and filtered games dfs
    merged_pbp = pd.merge(pbp_shots_df, filtered_games, on='game_id')

    if save_file is not None:
        merged_pbp.to_csv(save_file, index=False)

    return merged_pbp

def convert_pbp_range_to_shots_df(pbp_df_orig, save_file=True):
    """
    Convert the given play by play dataset into a shot result dataset.

    Args:
        pbp_df_orig (pd.DataFrame): Original play-by-play dataset.
        save_file (bool, optional): True if curated shot dataset should be saved to a CSV file;
         False otherwise. Defaults to True.

    Returns:
        pd.DataFrame: Shot result dataset.
    """
    pbp_df = pbp_df_orig.copy()

    # new column for fgm based on eventmsgtype
    pbp_df['fgm'] = pbp_df['eventmsgtype'].apply(lambda x: 1 if x == 1 else 0)

    # convert pctimestring to seconds_rem
    pbp_df['seconds_rem'] = pbp_df['pctimestring'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # rename player1_id to player_id and player1_team_id to team_id
    pbp_df.rename(columns={'player1_id': 'player_id', 'player1_team_id': 'team_id'}, inplace=True)

    # make IDs integers
    pbp_df[['player_id', 'team_id']] = pbp_df[['player_id', 'team_id']].astype('Int64')

    # add column for whether the shot was 3pt or not
    pbp_df['3pt'] = (
        pbp_df['homedescription'].str.contains('3PT', na=False) |
        pbp_df['visitordescription'].str.contains('3PT', na=False)
    ).astype(int)

    # select only the required columns
    columns_to_keep = ['game_id', 'game_date', 'eventnum', 'player_id', 'team_id', 'period', 'seconds_rem', '3pt', 'fgm']
    shots_df = pbp_df[columns_to_keep]

    if save_file:
        shots_df.to_csv(BASIC_SHOTS_FILE, index=False)

    return shots_df

def add_prev_shot_results(df_orig, num_shots=5, save_file=True):
    """
    Given a shot log dataset, add previous shot result columns. This includes a shot number
    column (grouped by game and player) to track the shot number indices for each shot a 
    player takes. It also includes fgm{i} and tot{i} columns which track the result of the
    ith previous shot and the number of made shots in the previous i shots, respectively. 

    Args:
        df_orig (pd.DataFrame): Shot log dataset
        num_shots (int, optional): Number of shots to include for shot history. Defaults to 5.
        save_file (bool, optional): True if curated shot history dataset should be saved to a CSV file;
         False otherwise. Defaults to True.

    Returns:
        pd.DataFrame: Shot history dataset
    """
    df = df_orig.copy()

    # shots need to be grouped by game and player (sort=False because df is already sorted in decreasing time)
    g = df.groupby(["game_id", "player_id"], sort=False)

    # add shot number column
    df["shot_number"] = g.cumcount() + 1

    for i in range(1, num_shots + 1):
        # add fgm{i} column (result of the shot taken i shots ago)
        df[f"fgm{i}"] = g["fgm"].shift(i)

        # add tot{i} column (result of last i shots, excluding current shot)
        df[f"tot{i}"] = g["fgm"].transform(
            lambda s: s.shift(1).rolling(i, min_periods=i).sum()
        )

    if save_file:
        df.to_csv(SHOT_HISTORY_FILE, index=False)
        
    return df


if __name__=='__main__':
    print('Creating play by play range dataset')
    pbp = create_pbp_range_dataset('2015-10-01', '2016-01-31', save_file=os.path.join('data', 'pbp_2015_2016.csv'))
    
    print('Converting pbp range dataset to shots dataset')
    shots_df = convert_pbp_range_to_shots_df(pbp)

    print('Creating shot history dataset')
    shot_history_df = add_prev_shot_results(shots_df, num_shots=10)