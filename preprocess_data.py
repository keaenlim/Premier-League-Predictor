import pandas as pd
from datetime import datetime

def preprocess_football_data(input_file='merged_data.csv', min_games_threshold=5):
    """
    Preprocess football match data:
    1. Assign unique IDs to teams
    2. Filter out matches with teams that don't have enough historical data
    
    Parameters:
    -----------
    input_file : str
        Path to the merged CSV file
    min_games_threshold : int
        Minimum number of games a team must have played before we predict their matches
    """
    
    # Read the data
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert Date to datetime for proper sorting
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Total matches in dataset: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Get all unique teams
    home_teams = set(df['HomeTeam'].unique())
    away_teams = set(df['AwayTeam'].unique())
    all_teams = sorted(home_teams.union(away_teams))
    
    print(f"\nTotal unique teams: {len(all_teams)}")
    print(f"Teams: {', '.join(all_teams)}")
    
    # Create team ID mapping
    team_to_id = {team: idx + 1 for idx, team in enumerate(all_teams)}
    id_to_team = {idx + 1: team for idx, team in enumerate(all_teams)}
    
    # Add team IDs to dataframe
    df['HomeTeamID'] = df['HomeTeam'].map(team_to_id)
    df['AwayTeamID'] = df['AwayTeam'].map(team_to_id)
    
    # Save team mapping
    team_mapping_df = pd.DataFrame([
        {'TeamID': team_id, 'TeamName': team_name} 
        for team_name, team_id in sorted(team_to_id.items())
    ])
    team_mapping_df.to_csv('team_mapping.csv', index=False)
    print(f"\nTeam mapping saved to team_mapping.csv")
    
    # Track games played by each team up to each point in time
    team_games_count = {team_id: 0 for team_id in team_to_id.values()}
    can_predict = []
    
    for idx, row in df.iterrows():
        home_id = row['HomeTeamID']
        away_id = row['AwayTeamID']
        
        # Check if both teams have enough historical data
        home_has_data = team_games_count[home_id] >= min_games_threshold
        away_has_data = team_games_count[away_id] >= min_games_threshold
        
        # We can predict this match if both teams have enough history
        can_predict.append(home_has_data and away_has_data)
        
        # Increment game count for both teams
        team_games_count[home_id] += 1
        team_games_count[away_id] += 1
    
    df['CanPredict'] = can_predict
    
    # Summary statistics
    total_matches = len(df)
    predictable_matches = df['CanPredict'].sum()
    filtered_matches = total_matches - predictable_matches
    
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY (min {min_games_threshold} games per team)")
    print(f"{'='*60}")
    print(f"Total matches: {total_matches}")
    print(f"Predictable matches: {predictable_matches}")
    print(f"Filtered out matches: {filtered_matches} ({filtered_matches/total_matches*100:.1f}%)")
    
    # Show which teams needed the most games filtered
    print(f"\nMatches filtered by season/team debut:")
    df_filtered = df[~df['CanPredict']].copy()
    df_filtered['Season'] = df_filtered['Date'].dt.year.astype(str) + '/' + (df_filtered['Date'].dt.year + 1).astype(str).str[-2:]
    
    for season in sorted(df_filtered['Season'].unique()):
        season_filtered = df_filtered[df_filtered['Season'] == season]
        teams_in_season = set(season_filtered['HomeTeam']).union(set(season_filtered['AwayTeam']))
        print(f"  {season}: {len(season_filtered)} matches filtered (teams: {', '.join(sorted(teams_in_season))})")
    
    # Create a summary showing team statistics (BEFORE converting dates back to strings)
    team_stats = []
    for team_name, team_id in team_to_id.items():
        home_matches = df[df['HomeTeamID'] == team_id]
        away_matches = df[df['AwayTeamID'] == team_id]
        total_matches = len(home_matches) + len(away_matches)
        
        # Find first appearance (while Date is still datetime)
        first_home = home_matches['Date'].min() if len(home_matches) > 0 else None
        first_away = away_matches['Date'].min() if len(away_matches) > 0 else None
        first_appearance = min([d for d in [first_home, first_away] if pd.notna(d)])
        
        # Format the date for display
        first_appearance_str = first_appearance.strftime('%d/%m/%Y')
        
        team_stats.append({
            'TeamID': team_id,
            'TeamName': team_name,
            'TotalMatches': total_matches,
            'HomeMatches': len(home_matches),
            'AwayMatches': len(away_matches),
            'FirstAppearance': first_appearance_str
        })
    
    team_stats_df = pd.DataFrame(team_stats).sort_values('TeamID')
    team_stats_df.to_csv('team_statistics.csv', index=False)
    print(f"Team statistics saved to team_statistics.csv")
    
    # Now convert dates back to string format for CSV output
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    df.to_csv('preprocessed_data_full.csv', index=False)
    print(f"\nFull preprocessed data saved to preprocessed_data_full.csv")
    
    # Save only predictable matches
    df_predictable = df[df['CanPredict']].copy()
    df_predictable.to_csv('preprocessed_data_predictable.csv', index=False)
    print(f"Predictable matches saved to preprocessed_data_predictable.csv")
    
    return df, team_to_id, id_to_team

if __name__ == "__main__":
    # Run preprocessing with default threshold of 5 games
    df, team_to_id, id_to_team = preprocess_football_data(
        input_file='merged_data.csv',
        min_games_threshold=5
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  1. team_mapping.csv - Maps team names to unique IDs")
    print("  2. team_statistics.csv - Summary stats for each team")
    print("  3. preprocessed_data_full.csv - All matches with team IDs and CanPredict flag")
    print("  4. preprocessed_data_predictable.csv - Only matches we can predict")
    print("\nUse 'preprocessed_data_predictable.csv' for model training!")
