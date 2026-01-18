import pandas as pd
import numpy as np
from collections import defaultdict

def engineer_features(input_file='preprocessed_data_predictable.csv', windows=[5, 10]):
    """
    Create rolling average features for each team based on their historical performance.
    
    Key principle: For each match, only use data from PREVIOUS matches (no data leakage).
    
    Parameters:
    input_file : str
        Path to preprocessed data with predictable matches only
    windows : list
        Rolling window sizes (e.g., [5, 10] means last 5 games and last 10 games)
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert Date to datetime for proper chronological ordering
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    '''
    print(f"Total matches to process: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Rolling windows: {windows}")
    '''
    
    # Initialize feature columns
    feature_cols = []
    for window in windows:
        for prefix in ['Home', 'Away']:
            # Goals features
            feature_cols.extend([
                f'{prefix}_GoalsScored_L{window}',
                f'{prefix}_GoalsConceded_L{window}',
                f'{prefix}_GoalDiff_L{window}'
            ])
            # Points features
            feature_cols.extend([
                f'{prefix}_Points_L{window}',
                f'{prefix}_WinRate_L{window}'
            ])
            # Shots features
            feature_cols.extend([
                f'{prefix}_ShotsOnTarget_L{window}',
                f'{prefix}_ShotsOnTargetAgainst_L{window}'
            ])
            # Defensive features
            feature_cols.append(f'{prefix}_CleanSheetRate_L{window}')
            # Draw and scoring pattern features
            feature_cols.extend([
                f'{prefix}_DrawRate_L{window}',
                f'{prefix}_LowScoringRate_L{window}',
                f'{prefix}_FailedToScoreRate_L{window}'
            ])

            # Shot conversion efficiency
            feature_cols.append(f'{prefix}_ConversionRate_L{window}')
    
    # Initialize all feature columns with NaN
    for col in feature_cols:
        df[col] = np.nan
    
    # Track rolling statistics for each team, separated by venue
    # Structure: team_history_home[team_id] = list of HOME match dictionaries
    #            team_history_away[team_id] = list of AWAY match dictionaries
    team_history_home = defaultdict(list)  # Team's performance when playing AT HOME
    team_history_away = defaultdict(list)  # Team's performance when playing AWAY
    
    print("\nCalculating rolling features chronologically...")
    
    # Process each match in chronological order
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing match {idx + 1}/{len(df)}...")
        
        home_id = row['HomeTeamID']
        away_id = row['AwayTeamID']
        
        # STEP 1: Calculate features for THIS match using PAST data only
        for team_id, is_home, prefix in [(home_id, True, 'Home'), (away_id, False, 'Away')]:
            # Use venue-specific history: home team's home games, away team's away games
            if is_home:
                team_past_matches = team_history_home[team_id]  # Home team: use their HOME history
            else:
                team_past_matches = team_history_away[team_id]  # Away team: use their AWAY history
            
            # Calculate rolling averages for each window
            for window in windows:
                # Get last N matches for this team
                recent_matches = team_past_matches[-window:] if len(team_past_matches) >= window else team_past_matches
                
                if len(recent_matches) > 0:
                    # Goals scored/conceded
                    goals_scored = np.mean([m['goals_scored'] for m in recent_matches])
                    goals_conceded = np.mean([m['goals_conceded'] for m in recent_matches])
                    goal_diff = goals_scored - goals_conceded
                    
                    # Points (3 for win, 1 for draw, 0 for loss)
                    points = np.mean([m['points'] for m in recent_matches])
                    
                    # Win rate
                    wins = sum(1 for m in recent_matches if m['result'] == 'W')
                    win_rate = wins / len(recent_matches)
                    
                    # Shots on target
                    shots_on_target = np.mean([m['shots_on_target'] for m in recent_matches])
                    shots_on_target_against = np.mean([m['shots_on_target_against'] for m in recent_matches])
                    
                    # Clean sheets
                    clean_sheets = sum(1 for m in recent_matches if m['goals_conceded'] == 0)
                    clean_sheet_rate = clean_sheets / len(recent_matches)

                    # Draw rate
                    draws = sum(1 for m in recent_matches if m['is_draw'] == 1)
                    draw_rate = draws / len(recent_matches)

                    # Low-scoring rate
                    low_scoring_matches = sum(1 for m in recent_matches if m['low_scoring'] == 1)
                    low_scoring_rate = low_scoring_matches / len(recent_matches)

                    # Failed to score rate
                    failed_to_score_matches = sum(1 for m in recent_matches if m['failed_to_score'] == 1)
                    failed_to_score_rate = failed_to_score_matches / len(recent_matches)

                    # Shot conversion rate (goals per shot on target)
                    total_shots = sum([m['shots_on_target'] for m in recent_matches])
                    total_goals = sum([m['goals_scored'] for m in recent_matches])

                    if total_shots > 0:
                        conversion_rate = total_goals / total_shots
                    else:
                        conversion_rate = 0.0  # Avoid division by zero

                    
                    # Assign to dataframe
                    df.at[idx, f'{prefix}_GoalsScored_L{window}'] = goals_scored
                    df.at[idx, f'{prefix}_GoalsConceded_L{window}'] = goals_conceded
                    df.at[idx, f'{prefix}_GoalDiff_L{window}'] = goal_diff
                    df.at[idx, f'{prefix}_Points_L{window}'] = points
                    df.at[idx, f'{prefix}_WinRate_L{window}'] = win_rate
                    df.at[idx, f'{prefix}_ShotsOnTarget_L{window}'] = shots_on_target
                    df.at[idx, f'{prefix}_ShotsOnTargetAgainst_L{window}'] = shots_on_target_against
                    df.at[idx, f'{prefix}_CleanSheetRate_L{window}'] = clean_sheet_rate
                    df.at[idx, f'{prefix}_DrawRate_L{window}'] = draw_rate
                    df.at[idx, f'{prefix}_LowScoringRate_L{window}'] = low_scoring_rate
                    df.at[idx, f'{prefix}_FailedToScoreRate_L{window}'] = failed_to_score_rate
                    df.at[idx, f'{prefix}_ConversionRate_L{window}'] = conversion_rate
        
        # STEP 2: After calculating features, add THIS match to team histories
        # (so it's available for FUTURE matches)
        
        # Home team's perspective
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        home_result = 'W' if row['FTR'] == 'H' else ('D' if row['FTR'] == 'D' else 'L')
        away_result = 'W' if row['FTR'] == 'A' else ('D' if row['FTR'] == 'D' else 'L')
        home_points = 3 if home_result == 'W' else (1 if home_result == 'D' else 0)
        away_points = 3 if away_result == 'W' else (1 if away_result == 'D' else 0)
        
        # Record home team's match in their HOME history (they played at home)
        team_history_home[home_id].append({
            'goals_scored': home_goals,
            'goals_conceded': away_goals,
            'result': home_result,
            'points': home_points,
            'shots_on_target': row['HST'],
            'shots_on_target_against': row['AST'],
            'is_draw': 1 if home_result == 'D' else 0,
            'low_scoring': 1 if (home_goals + away_goals) <= 2 else 0,
            'failed_to_score': 1 if home_goals == 0 else 0
        })
        
        # Record away team's match in their AWAY history (they played away)
        team_history_away[away_id].append({
            'goals_scored': away_goals,
            'goals_conceded': home_goals,
            'result': away_result,
            'points': away_points,
            'shots_on_target': row['AST'],
            'shots_on_target_against': row['HST'],
            'is_draw': 1 if away_result == 'D' else 0,
            'low_scoring': 1 if (home_goals + away_goals) <= 2 else 0,
            'failed_to_score': 1 if away_goals == 0 else 0
        })
    
    # STEP 3: Calculate matchup comparison features
    # These require BOTH home and away features to exist first
    
    print("\nCalculating matchup comparison features...")
    
    for window in windows:
        # FEATURE 3: Strength similarity gaps (smaller = more evenly matched = draw likely)
        
        # Goal difference gap
        df[f'GoalDiff_Gap_L{window}'] = abs(
            df[f'Home_GoalDiff_L{window}'] - df[f'Away_GoalDiff_L{window}']
        )
        
        # Points gap
        df[f'Points_Gap_L{window}'] = abs(
            df[f'Home_Points_L{window}'] - df[f'Away_Points_L{window}']
        )
        
        # Shot quality gap
        df[f'ShotsOnTarget_Gap_L{window}'] = abs(
            df[f'Home_ShotsOnTarget_L{window}'] - df[f'Away_ShotsOnTarget_L{window}']
        )
        
        # Conversion rate gap (similar finishing ability = draw likely)
        df[f'ConversionRate_Gap_L{window}'] = abs(
            df[f'Home_ConversionRate_L{window}'] - df[f'Away_ConversionRate_L{window}']
        )
        
        # FEATURE 4: Combined defensive & draw tendency (high values = draw likely)
        
        # Combined defensive strength (both teams defensive = low-scoring draw)
        df[f'Combined_CleanSheetRate_L{window}'] = (
            df[f'Home_CleanSheetRate_L{window}'] + df[f'Away_CleanSheetRate_L{window}']
        ) / 2
        
        # Combined draw tendency (both teams draw often)
        df[f'Combined_DrawRate_L{window}'] = (
            df[f'Home_DrawRate_L{window}'] + df[f'Away_DrawRate_L{window}']
        ) / 2
        
        # Combined low-scoring tendency
        df[f'Combined_LowScoringRate_L{window}'] = (
            df[f'Home_LowScoringRate_L{window}'] + df[f'Away_LowScoringRate_L{window}']
        ) / 2
        
        # Combined failed to score rate
        df[f'Combined_FailedToScoreRate_L{window}'] = (
            df[f'Home_FailedToScoreRate_L{window}'] + df[f'Away_FailedToScoreRate_L{window}']
        ) / 2
    
    print("✓ Matchup comparison features calculated")
    
    # Check how many rows have complete features
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    # Count rows with all features populated
    sample_feature = f'Home_GoalsScored_L{windows[0]}'
    complete_rows = df[sample_feature].notna().sum()
    print(f"Matches with complete features: {complete_rows}/{len(df)}")
    print(f"Matches with missing features: {len(df) - complete_rows}")
    
    # Show sample of engineered features
    print(f"\nSample of engineered features (first complete match):")
    first_complete_idx = df[sample_feature].notna().idxmax()
    sample_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR'] + feature_cols[:8]
    print(df.loc[first_complete_idx, sample_cols].to_string())
    
    # Convert date back to string for CSV output
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    
    # Save the engineered dataset
    output_file = 'data_with_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Engineered data saved to {output_file}")
    
    # Also save a version with only complete rows (ready for ML)
    df_complete = df[df[sample_feature].notna()].copy()
    output_file_complete = 'data_with_features_complete.csv'
    df_complete.to_csv(output_file_complete, index=False)
    print(f"✓ Complete data (no missing features) saved to {output_file_complete}")
    print(f"  → {len(df_complete)} matches ready for model training")
    
    return df, df_complete

if __name__ == "__main__":
    # Run feature engineering with 5 and 10 game windows
    df_all, df_complete = engineer_features(
        input_file='preprocessed_data_predictable.csv',
        windows=[5, 10]
    )
    
    '''
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Use 'data_with_features_complete.csv' for model training")
    print("2. Features include rolling averages over 5 and 10 game windows")
    print("3. You can easily add more features by modifying this script")
    print("="*60)
    '''
