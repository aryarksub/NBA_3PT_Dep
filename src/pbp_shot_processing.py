import pandas as pd
import os

from moment_processing import MOMENT_DATA_DIR

# PBP data source: https://www.kaggle.com/datasets/wyattowalsh/basketball?resource=download
FULL_PBP_FILE = os.path.join('data', 'play_by_play.csv')
GAME_INFO_FILE = os.path.join('data', 'game_info.csv')
BASIC_SHOTS_FILE = os.path.join('data', 'pbp_shots_basic.csv')

def get_stored_moment_game_ids():
    """
    Get the game IDs for all games with moment data that have been stored

    Returns:
        list: List of stored game ID strings
    """
    return [file.split('.')[0][2:] for root, dirs, files in os.walk(MOMENT_DATA_DIR) for file in files]

def create_pbp_range_dataset(start_date, end_date, exclude_non_stored=True):
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


if __name__=='__main__':
    pbp = create_pbp_range_dataset('2015-10-01', '2016-01-31')
    shots_df = convert_pbp_range_to_shots_df(pbp)