import os
import numpy as np
import pandas as pd

def read_team_data(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data'):
    
    seasons = [season for season in os.listdir(path) if os.path.isdir(os.path.join(path, season))]

    team_dfs = []
   
    for season in seasons:
        if(season in ["2018-19","2017-18","2016-17"]):
            continue
        team_csv_path = os.path.join(path,season,"teams.csv")
        team_df = pd.read_csv(team_csv_path)
        team_df["season"] = season

        team_dfs.append(team_df)

    team_data = pd.concat(team_dfs,ignore_index=True)
    
    return team_data

def read_player_data(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data'):

    player_df = pd.read_csv(path + r'\cleaned_merged_seasons.csv')
    player_df = player_df[~player_df.season_x.isin(['2016-17','2017-18','2018-19','2019-20'])]
    player_df = player_df[~player_df.team_x.isnull()]
    return player_df

def translate_name(name):
    if "_" in name:
        return name.split("_")[0] + " " + name.split("_")[1]
    else:
        return name

def read_gw_data(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data'):

    seasons = [season for season in os.listdir(path) if os.path.isdir(os.path.join(path, season))]

    team_dfs = []

    for season in seasons:
        if(season in ["2018-19","2017-18","2016-17"]):
            continue
        team_csv_path = os.path.join(path,season,"gws","merged_gw.csv")
        team_df = pd.read_csv(team_csv_path)
        team_df["season"] = season

        team_dfs.append(team_df)

    team_data = pd.concat(team_dfs,ignore_index=True)
    team_data['kickoff_time'] = pd.to_datetime(team_data['kickoff_time'],utc=True)
    team_data['name'] = team_data['name'].apply(translate_name)
    return team_data


def read_player_gw_data(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data'):
    seasons = [season for season in os.listdir(path) if os.path.isdir(os.path.join(path, season))]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Iterate through each season
    for season in seasons:
        # Construct the path to the players directory for the current season
        players_dir = os.path.join(path, season, "players")
        
        # Get the list of player directories for the current season
        players = [player for player in os.listdir(players_dir) if os.path.isdir(os.path.join(players_dir, player))]
        
        # Iterate through each player
        for player in players:
            if(player =='players'):
                continue
            # Construct the path to the player's gw.csv file
            gw_csv_path = os.path.join(players_dir, player, "gw.csv")
            name = player.split('_')[0] + (' ') + player.split('_')[1]
            # Read the gw.csv file into a dataframe
            if(os.path.exists(gw_csv_path)):
                df = pd.read_csv(gw_csv_path)
                
                # Add a 'season' column with the current season value
                df['season'] = season
                df['player'] = name
                # Append the dataframe to the list
                dataframes.append(df)
    # Concatenate all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df[['player','assists','bonus','bps','clean_sheets','creativity','expected_assists','expected_goal_involvements',
    'expected_goals','expected_goals_conceded','goals_conceded','goals_scored','kickoff_time','ict_index','influence','minutes',
    'own_goals','penalties_missed','penalties_saved','red_cards','saves','selected','starts','team_a_score','team_h_score','threat','total_points','was_home','yellow_cards','fixture', 'season']]
    combined_df['kickoff_time'] = pd.to_datetime(combined_df['kickoff_time'],utc=True)
    return combined_df
 
def exponential_decay(t,decay_rate):
    return np.exp(-decay_rate*t)

def normalise_for_match(df):
    new_df = df.copy()
    for col in new_df.columns:
        if col in ['goals_scored', 'assists', 'goals_conceded', 'own_goals',
       'penalties_missed', 'penalties_saved', 'saves']:
            new_df[col] = new_df[col] * 90 / new_df['minutes']
    return new_df

def player_profile(player,kickoff_time,players_history,decay_rate=0.02):
    
    profile_columns = ['assists','bonus','bps','clean_sheets','creativity','goals_conceded','goals_scored','ict_index','influence','minutes',
    'own_goals','penalties_missed','penalties_saved','red_cards','saves','selected','team_a_score','team_h_score','threat','total_points','was_home','yellow_cards']
    player_history = players_history[players_history['player']==player]
    player_history_len = len(player_history)
    player_history = player_history[pd.to_datetime(player_history['kickoff_time']) < pd.Timestamp(kickoff_time)]
    if(len(player_history)==0):
        return dict.fromkeys(profile_columns,0)
    player_history['weight'] = exponential_decay(pd.to_timedelta( pd.Timestamp(kickoff_time) - pd.to_datetime(player_history['kickoff_time']),unit='D') / np.timedelta64(1, 'D'),decay_rate)
    player_history = player_history.drop(['fixture','season','kickoff_time'],axis=1)
    
    
    player_profile  = {}
    for col in profile_columns:
        player_profile['profile_'+col] = np.average(player_history[col],weights=player_history['weight'])

    return player_profile

def augment_fixtures_player_profiles(fixtures_input,players,teams,decay_rate=0.02):

    fixtures = fixtures_input.sort_values(by=['name','kickoff_time'])
    duplicate_players = fixtures['name'].duplicated(keep='first')
    fixtures = fixtures[duplicate_players]
    def calculate_player_profile(row):
        return player_profile(row['name'],row['kickoff_time'],players,decay_rate)

    new_columns = fixtures.progress_apply(lambda row: calculate_player_profile(row), axis = 1)
    new_columns_df = pd.DataFrame(new_columns.to_list(),index=fixtures.index)
    result_df = pd.concat([fixtures,new_columns_df],axis=1)
    return result_df
    
def read_pre_20_fixtures(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data'):   
    seasons = [season for season in os.listdir(path) if os.path.isdir(os.path.join(path, season))]

    team_dfs = []

    for season in seasons:
        if(season not in ["2018-19","2017-18"]):
            continue
        team_csv_path = os.path.join(path,season,"fixtures.csv")
        team_df = pd.read_csv(team_csv_path)
        team_df["season"] = season

        team_dfs.append(team_df)

    team_data = pd.concat(team_dfs,ignore_index=True)
    team_data['kickoff_time'] = pd.to_datetime(team_data['kickoff_time'],utc=True)
    
    return team_data

def read_season_fixtures_data(path = r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data',season='2023-24'):
    fixtures = pd.read_csv(os.path.join(path,season,'fixtures.csv'))
    fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time'],utc=True)
    return fixtures

def read_season_players_data(path= r'C:\Users\charles.fletcher\Documents\Repos\fpl\fpl-predictor\data',season='2023-24'):
    players = pd.read_csv(os.path.join(path,season,'players_raw.csv'))
    return players