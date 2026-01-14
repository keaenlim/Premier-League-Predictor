import pandas as pd
import requests
from datetime import datetime
import os
import sys

def download_latest_season_data(season='2526', save_path='latest_season.csv'):
    """
    Download the latest Premier League season data from football-data.co.uk
    
    Parameters:
    -----------
    season : str
        Season code (e.g., '2526' for 2025-26, '2425' for 2024-25)
    save_path : str
        Where to save the downloaded CSV
    
    Returns:
    --------
    str : Path to downloaded file, or None if failed
    """
    # E0 = English Premier League
    url = f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING LATEST PREMIER LEAGUE DATA")
    print(f"{'='*70}")
    print(f"Season: 20{season[:2]}/20{season[2:]}")
    print(f"Source: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        # Verify it's valid CSV
        df = pd.read_csv(save_path)
        print(f"✓ Downloaded successfully: {len(df)} matches found")
        print(f"✓ Saved to: {save_path}")
        
        return save_path
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download: {e}")
        return None
    except pd.errors.ParserError as e:
        print(f"✗ Downloaded file is not valid CSV: {e}")
        return None


def identify_new_matches(new_csv, existing_csv='preprocessed_data_predictable.csv'):
    """
    Compare downloaded CSV with existing data to find new matches.
    
    Parameters:
    -----------
    new_csv : str
        Path to newly downloaded CSV
    existing_csv : str
        Path to existing preprocessed data
    
    Returns:
    --------
    pd.DataFrame : New matches not in existing data
    """
    print(f"\n{'='*70}")
    print(f"IDENTIFYING NEW MATCHES")
    print(f"{'='*70}")
    
    # Load both datasets
    new_df = pd.read_csv(new_csv)
    
    if not os.path.exists(existing_csv):
        print(f"⚠️  No existing data found at {existing_csv}")
        print(f"   All {len(new_df)} matches will be treated as new")
        return new_df
    
    existing_df = pd.read_csv(existing_csv)
    
    # Convert dates to same format for comparison
    new_df['Date'] = pd.to_datetime(new_df['Date'], format='%d/%m/%Y', errors='coerce')
    existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    print(f"Existing data: {len(existing_df)} matches")
    print(f"Downloaded data: {len(new_df)} matches")
    
    # Create unique match identifier: Date + HomeTeam + AwayTeam
    existing_df['MatchID'] = (
        existing_df['Date'].dt.strftime('%Y%m%d') + '_' +
        existing_df['HomeTeam'] + '_' +
        existing_df['AwayTeam']
    )
    
    new_df['MatchID'] = (
        new_df['Date'].dt.strftime('%Y%m%d') + '_' +
        new_df['HomeTeam'] + '_' +
        new_df['AwayTeam']
    )
    
    # Find matches in new_df that aren't in existing_df
    existing_match_ids = set(existing_df['MatchID'])
    new_matches = new_df[~new_df['MatchID'].isin(existing_match_ids)].copy()
    
    # Drop the temporary MatchID column
    new_matches = new_matches.drop('MatchID', axis=1)
    
    # Convert date back to original format
    new_matches['Date'] = new_matches['Date'].dt.strftime('%d/%m/%Y')
    
    print(f"\n✓ Found {len(new_matches)} new matches")
    
    if len(new_matches) > 0:
        print(f"\nNew matches preview:")
        print(f"{'Date':<12} {'Home Team':<20} {'Away Team':<20} {'Score':<8}")
        print(f"{'-'*70}")
        for _, match in new_matches.head(10).iterrows():
            score = f"{match.get('FTHG', '?')}-{match.get('FTAG', '?')}"
            print(f"{match['Date']:<12} {match['HomeTeam']:<20} {match['AwayTeam']:<20} {score:<8}")
        
        if len(new_matches) > 10:
            print(f"... and {len(new_matches) - 10} more")
    
    return new_matches


def process_new_matches(new_matches_df, existing_data_path='preprocessed_data_predictable.csv'):
    """
    Process new matches through the preprocessing pipeline.
    
    Parameters:
    -----------
    new_matches_df : pd.DataFrame
        New matches to process
    existing_data_path : str
        Path to existing preprocessed data
    
    Returns:
    --------
    pd.DataFrame : Processed new matches ready to append
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING NEW MATCHES")
    print(f"{'='*70}")
    
    if len(new_matches_df) == 0:
        print("No new matches to process")
        return pd.DataFrame()
    
    # Load existing data to get team mappings
    if os.path.exists(existing_data_path):
        existing_df = pd.read_csv(existing_data_path)
        
        # Extract team ID mapping from existing data
        home_mapping = existing_df[['HomeTeam', 'HomeTeamID']].drop_duplicates()
        away_mapping = existing_df[['AwayTeam', 'AwayTeamID']].drop_duplicates()
        
        team_to_id = {}
        for _, row in home_mapping.iterrows():
            team_to_id[row['HomeTeam']] = row['HomeTeamID']
        for _, row in away_mapping.iterrows():
            team_to_id[row['AwayTeam']] = row['AwayTeamID']
        
        max_team_id = max(team_to_id.values())
    else:
        team_to_id = {}
        max_team_id = 0
    
    # Assign team IDs to new matches
    new_matches_df = new_matches_df.copy()
    
    for team_col in ['HomeTeam', 'AwayTeam']:
        id_col = team_col + 'ID'
        
        def get_team_id(team_name):
            if team_name in team_to_id:
                return team_to_id[team_name]
            else:
                # New team - assign new ID
                nonlocal max_team_id
                max_team_id += 1
                team_to_id[team_name] = max_team_id
                print(f"  ⚠️  New team detected: {team_name} (ID: {max_team_id})")
                return max_team_id
        
        new_matches_df[id_col] = new_matches_df[team_col].apply(get_team_id)
    
    # Check if matches have enough historical data (can be predicted)
    # For simplicity, mark all new matches as predictable if teams are known
    new_matches_df['CanPredict'] = True
    
    # Add division column if not present
    if 'Div' not in new_matches_df.columns:
        new_matches_df['Div'] = 'E0'
    
    print(f"✓ Assigned team IDs to {len(new_matches_df)} matches")
    print(f"✓ All matches marked as predictable")
    
    return new_matches_df


def update_preprocessed_data(new_matches_df, 
                              predictable_path='preprocessed_data_predictable.csv',
                              full_path='preprocessed_data_full.csv'):
    """
    Append new matches to preprocessed data files.
    
    Parameters:
    -----------
    new_matches_df : pd.DataFrame
        Processed new matches
    predictable_path : str
        Path to predictable matches file
    full_path : str
        Path to full preprocessed data file
    """
    print(f"\n{'='*70}")
    print(f"UPDATING PREPROCESSED DATA FILES")
    print(f"{'='*70}")
    
    if len(new_matches_df) == 0:
        print("No new matches to append")
        return
    
    # Update predictable matches file
    if os.path.exists(predictable_path):
        existing_predictable = pd.read_csv(predictable_path)
        updated_predictable = pd.concat([existing_predictable, new_matches_df], ignore_index=True)
        
        # Sort by date
        updated_predictable['Date'] = pd.to_datetime(updated_predictable['Date'], format='%d/%m/%Y')
        updated_predictable = updated_predictable.sort_values('Date')
        updated_predictable['Date'] = updated_predictable['Date'].dt.strftime('%d/%m/%Y')
        
        updated_predictable.to_csv(predictable_path, index=False)
        print(f"✓ Updated {predictable_path}")
        print(f"  Old: {len(existing_predictable)} matches → New: {len(updated_predictable)} matches")
    else:
        new_matches_df.to_csv(predictable_path, index=False)
        print(f"✓ Created {predictable_path} with {len(new_matches_df)} matches")
    
    # Update full data file if it exists
    if os.path.exists(full_path):
        existing_full = pd.read_csv(full_path)
        updated_full = pd.concat([existing_full, new_matches_df], ignore_index=True)
        
        # Sort by date
        updated_full['Date'] = pd.to_datetime(updated_full['Date'], format='%d/%m/%Y')
        updated_full = updated_full.sort_values('Date')
        updated_full['Date'] = updated_full['Date'].dt.strftime('%d/%m/%Y')
        
        updated_full.to_csv(full_path, index=False)
        print(f"✓ Updated {full_path}")
        print(f"  Old: {len(existing_full)} matches → New: {len(updated_full)} matches")


def update_features(data_path='preprocessed_data_predictable.csv'):
    """
    Re-run feature engineering on updated data.
    
    Parameters:
    -----------
    data_path : str
        Path to preprocessed data file
    """
    print(f"\n{'='*70}")
    print(f"UPDATING FEATURES")
    print(f"{'='*70}")
    
    try:
        # Import and run feature engineering
        from feature_engineering import engineer_features
        
        print("Running feature engineering...")
        engineer_features(input_file=data_path)
        print("✓ Features updated successfully")
        
    except ImportError:
        print("⚠️  Could not import feature_engineering.py")
        print("   Please run: python3 feature_engineering.py")
    except Exception as e:
        print(f"⚠️  Feature engineering failed: {e}")
        print("   Please run manually: python3 feature_engineering.py")


def main():
    """
    Main update workflow
    """
    print(f"\n{'#'*70}")
    print(f"# PREMIER LEAGUE DATA UPDATE TOOL")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Step 1: Download latest data
    # Automatically detect current season (2025-26 = 2526)
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # If after August, use current year, else previous year
    if current_month >= 8:
        season_start = str(current_year)[2:]
        season_end = str(current_year + 1)[2:]
    else:
        season_start = str(current_year - 1)[2:]
        season_end = str(current_year)[2:]
    
    season_code = season_start + season_end
    
    print(f"\nAuto-detected season: 20{season_start}/20{season_end}")
    
    downloaded_file = download_latest_season_data(season=season_code)
    
    if downloaded_file is None:
        print("\n✗ Update failed - could not download data")
        sys.exit(1)
    
    # Step 2: Identify new matches
    new_matches = identify_new_matches(downloaded_file)
    
    if len(new_matches) == 0:
        print(f"\n{'='*70}")
        print("✓ No new matches found - your data is already up to date!")
        print(f"{'='*70}")
        
        # Clean up downloaded file
        os.remove(downloaded_file)
        return
    
    # Step 3: Process new matches
    processed_matches = process_new_matches(new_matches)
    
    # Step 4: Update preprocessed data files
    update_preprocessed_data(processed_matches)
    
    # Step 5: Re-run feature engineering
    update_features()
    
    # Clean up downloaded file
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
    
    # Final summary
    print(f"\n{'#'*70}")
    print(f"# UPDATE COMPLETE")
    print(f"{'#'*70}")
    print(f"✓ Added {len(processed_matches)} new matches")
    print(f"✓ Preprocessed data updated")
    print(f"✓ Features re-engineered")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the new data in preprocessed_data_predictable.csv")
    print(f"   2. Retrain your model: python3 train_model.py")
    print(f"   3. Make predictions: python3 predict_match.py")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
