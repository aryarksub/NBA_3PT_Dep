import pandas as pd
import numpy as np
import os
from scipy.spatial import ConvexHull

from moment_processing import MOMENT_DATA_DIR

# PBP data source: https://www.kaggle.com/datasets/wyattowalsh/basketball?resource=download
FULL_PBP_FILE = os.path.join('data', 'play_by_play.csv')
GAME_INFO_FILE = os.path.join('data', 'game_info.csv')
BASIC_SHOTS_FILE = os.path.join('data', 'pbp_shots_basic.csv')
SHOT_HISTORY_FILE = os.path.join('data', 'pbp_shots_history.csv')
SHOT_HISTORY_DEF_FILE = os.path.join('data', 'pbp_shots_history_with_def.csv')

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

    # rename columns
    pbp_df.rename(columns={'player1_id': 'player_id', 'player1_team_id': 'team_id', 'eventnum': 'event_id'}, inplace=True)

    # make IDs integers
    pbp_df[['player_id', 'team_id']] = pbp_df[['player_id', 'team_id']].astype('Int64')

    # add column for whether the shot was 3pt or not
    pbp_df['3pt'] = (
        pbp_df['homedescription'].str.contains('3PT', na=False) |
        pbp_df['visitordescription'].str.contains('3PT', na=False)
    ).astype(int)

    # select only the required columns
    columns_to_keep = ['game_id', 'game_date', 'event_id', 'player_id', 'team_id', 'period', 'seconds_rem', '3pt', 'fgm']
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
    We also add a streak column that tracks the number of consecutive made shots before the
    current shot was taken.

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

    def compute_streak(s):
        """
        Compute the streak of consecutive successful shot makes.

        Args:
            s (pd.Series): Series of field goal results (0 = miss, 1 = make).

        Returns:
            pd.Series: Series where each value represents the current streak of consecutive shot makes for that 
            position. Streak resets to 0 after a miss.
        """
        # Shift to compare shots to previous shot
        prev = s.shift(1)
        # Group shot makes by checking when 1 turns to 0 (miss after make)
        streak = prev.groupby((prev != 1).cumsum()).cumcount()
        # Reset streak to 0 whenever there's a miss (prev != 1)
        streak[prev != 1] = 0
        return streak
    
    df['streak'] = g['fgm'].transform(compute_streak)

    if save_file:
        df.to_csv(SHOT_HISTORY_FILE, index=False)
        
    return df

def find_release_row(moments_df, player_id, target_clock):
    """
    Find the exact "release row" (moment in the moments DataFrame) that the shot is taken. This is
    defined by the earliest time before the target clock in which the given player has possesion and
    the ball height is monotonic increasing afterwards. 

    Args:
        moments_df (pd.DataFrame): DataFrame of moments
        player_id (int): Unique ID of player that took the shot
        target_clock (float): Number of seconds left when the shot was taken, as reported by
         play-by-play data. This might not accurately represent the actual release time of 
         the shot which is why scanning the moments DataFrame is needed.

    Returns:
        pd.Series: Moment row corresponding to release time
    """
    # Find starting index closest in time
    start_idx = (moments_df["clock"] - target_clock).abs().idxmin()

    # Work backwards from this index
    subset = moments_df.loc[:start_idx].iloc[::-1]

    # Part 1: find most recent shooter-possession frame
    in_possession = False
    possession_rows = []

    for _, row in subset.iterrows():
        if row.closest_player_id1 == player_id:
            in_possession = True
            possession_rows.append(row)
        # Shooter possession window ended
        elif in_possession:
            break

    if not possession_rows:
        return None

    # possession_rows are in reverse time order ... put in chronological order
    possession_rows = possession_rows[::-1]

    # Part 2: find ball_z takeoff point (latest time within possession window such that
    # ball height is monotonic increasing afterwards)
    min_future_z = possession_rows[-1].ball_z
    takeoff_row = possession_rows[-1]

    for row in reversed(possession_rows[:-1]):
        if row.ball_z < min_future_z:
            min_future_z = row.ball_z
            takeoff_row = row
        else:
            break

    return takeoff_row

def compute_shot_def_info_from_moment(moment_row, player_id):
    """
    Compute shot and defender information for the shot corresponding to the given moment.
    This includes the shot distance, time left on the shot clock, closest defender player ID,
    closest defender's team ID, closest defender's distance to shooter, average defender
    distance, and convex hull area of all defenders.

    Args:
        moment_row (pd.Series): Moment data for the shot
        player_id (int): Unique ID for the shooter
    """
    def get_shooter_index(data):
        """
        Get the index of the shooter (0 to 9) in the given moment data dictionary.

        Args:
            data (dict): Moment data stored as a dict

        Returns:
            int: Shooter index (0 to 9)
        """
        for i in range(10):
            if data[f'player{i}_id'] == player_id:
                return i
        return None
    
    def get_defender_info(data, shooter_index):
        """
        Get the defender information from the moment data given the shooter's index.

        Args:
            data (dict): Moment data stored as a dict
            shooter_index (int): Shooter index (0 to 9)

        Returns:
            list: List of defender information
        """
        defenders = []
        shooter_x, shooter_y = data[f'player{shooter_index}_x'], data[f'player{shooter_index}_y']
        for i in range(10):
            # Defenders are players who are on different team than shooter
            if data[f'player{i}_team_id'] != shooter_team_id:
                def_x, def_y = data[f'player{i}_x'], data[f'player{i}_y']
                def_dist = np.sqrt((def_x - shooter_x)**2 + (def_y - shooter_y)**2)
                # Defender info: ID, team ID, x-coord, y-coord, distance to shooter
                defenders.append([
                    data[f'player{i}_id'], data[f'player{i}_team_id'], def_x, def_y, def_dist
                ])
        assert len(defenders) == 5

        # Sort defenders in increasing order of distance to shooter
        defenders.sort(key = (lambda x : x[-1]))
        return defenders

    def compute_shot_dist(x, y):
        """
        Compute the distance from the player (x,y) to the hoop (effectively measures
        shot distance).

        Args:
            x (float): Player x-coordinate
            y (float): Player y-coordinate

        Returns:
            float: Shot distance
        """
        # Basket is 5 feet in front of each baseline (94 feet baseline)
        left_hoop = np.array([5, 25])
        right_hoop = np.array([89, 25])
        player_loc = np.array([x, y])
        # Return the shorter distance between player to left hoop and player to right hoop
        return min(np.linalg.norm(player_loc - left_hoop), np.linalg.norm(player_loc - right_hoop))
    
    def compute_convex_hull_area(def_info):
        """
        Compute the area of the convex hull formed from defender locations using the 
        full defender information list.

        Args:
            def_info (list): List of defender information

        Returns:
            float: Defender convex hull area
        """
        # defender locations (x,y) are stored in indices 2,3
        def_locs = [[data[2], data[3]] for data in def_info]
        hull = ConvexHull(def_locs)
        return hull.volume
    
    # Convert pd.Series (moment_row) to dictionary
    moment_data = moment_row.to_dict()
    # Shooter index (0 to 9)
    shooter_index = get_shooter_index(moment_data)
    # Shooter information
    shooter_team_id = moment_data[f'player{shooter_index}_team_id']
    shooter_x, shooter_y = moment_data[f'player{shooter_index}_x'], moment_data[f'player{shooter_index}_y']
    # Defender information and metrics
    defenders = get_defender_info(moment_data, shooter_index)
    avg_def_dist = sum(defender[-1] for defender in defenders) / len(defenders)
    def_hull_area = compute_convex_hull_area(defenders)

    return (
        compute_shot_dist(shooter_x, shooter_y), 
        moment_data['shot_clock'], 
        defenders[0][0], 
        defenders[0][1], 
        defenders[0][-1], 
        avg_def_dist, 
        def_hull_area
    )

def add_shot_def_info(shot_df_orig, save_file=True):
    """
    Given a shot log dataset, add columns for shot and defender information. This includes 
    the shot distance, time left on the shot clock, closest defender player ID, closest 
    defender's team ID, closest defender's distance to shooter, average defender distance, 
    and convex hull area of all defenders.

    Args:
        shot_df_orig (pd.DataFrame): Shot log DataFrame
        save_file (bool, optional): True if curated dataset should be saved to a CSV file;
         False otherwise. Defaults to True.

    Returns:
        pd.DataFrame: Shot log dataset with shot and defender information
    """
    shot_df = shot_df_orig.copy()
    new_cols = ["shot_dist", "shot_clock", "close_def_id", "close_def_team_id", "close_def_dist", "avg_def_dist", "def_hull_area"]
    shot_df[new_cols] = np.nan

    # Group shots by game
    for game_id, shots_g in shot_df.groupby("game_id"):
        moments_df = pd.read_csv(os.path.join(MOMENT_DATA_DIR, f'00{game_id}.csv'))
        moments_df = moments_df.sort_values(
            ["event_id", "period", "clock"],
            ascending=[True, True, False]
        )
        
        for row in shots_g.itertuples():
            # DF of moments that match the same period as the shot
            period_subset = moments_df[
                (moments_df.period == row.period)
            ]
            if period_subset.shape[0] == 0:
                continue

            # Find row in moment DF where the ball was released
            release_moment_row = find_release_row(period_subset, row.player_id, row.seconds_rem)

            # Some moments are missing, so if the release row cannot be found, ignore the entire shot
            if release_moment_row is None:
                continue

            # Compute shot clock and defender information
            clock_def_info = compute_shot_def_info_from_moment(release_moment_row, row.player_id)

            # Store defender information + shot clock
            shot_df.loc[row.Index, new_cols] = clock_def_info

    if save_file:
        shot_df.to_csv(SHOT_HISTORY_DEF_FILE, index=False)

    return shot_df


if __name__=='__main__':
    print('Creating play by play range dataset')
    pbp = create_pbp_range_dataset('2015-10-01', '2016-01-31', save_file=os.path.join('data', 'pbp_2015_2016.csv'))
    
    print('Converting pbp range dataset to shots dataset')
    shots_df = convert_pbp_range_to_shots_df(pbp)

    print('Creating shot history dataset')
    shot_history_df = add_prev_shot_results(shots_df, num_shots=10)

    print('Creating shot history dataset with defender information')
    shot_history_def_df = add_shot_def_info(shot_history_df)