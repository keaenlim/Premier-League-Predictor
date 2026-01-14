"""
Helper script to show all available teams in the dataset.
Useful for knowing the exact team name spellings to use in predictions.
"""

import pandas as pd

# Load the data
df = pd.read_csv('data_with_features_complete.csv')

# Get unique team names
home_teams = set(df['HomeTeam'].unique())
away_teams = set(df['AwayTeam'].unique())
all_teams = sorted(home_teams.union(away_teams))

print("\n" + "="*70)
print("AVAILABLE TEAMS IN DATASET")
print("="*70)
print(f"\nTotal Teams: {len(all_teams)}\n")

for i, team in enumerate(all_teams, 1):
    # Count how many matches this team has played
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
    num_matches = len(team_matches)
    
    # Get date of most recent match
    team_matches['Date'] = pd.to_datetime(team_matches['Date'], format='%d/%m/%Y')
    most_recent = team_matches['Date'].max().strftime('%d/%m/%Y')
    
    print(f"{i:2d}. {team:<20} ({num_matches:3d} matches, last: {most_recent})")

print("\n" + "="*70)
print("USAGE TIPS:")
print("="*70)
print("1. Copy the exact team name spelling from above")
print("2. Use it in predict_match():")
print("   result = predict_match(")
print("       home_team='Arsenal',  # Exact spelling!")
print("       away_team='Liverpool',")
print("       betting_odds={...}")
print("   )")
print("="*70 + "\n")
