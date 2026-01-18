import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime

def get_team_recent_stats(team_name, data_file='data_with_features_complete.csv', num_games=10):
    """
    Extract recent statistics for a team from historical data.
    
    Parameters:
    -----------
    team_name : str
        Name of the team (e.g., 'Arsenal', 'Liverpool')
    data_file : str
        Path to CSV file containing historical match data
    num_games : int
        Number of recent games to analyze (5 or 10)
    
    Returns:
    --------
    dict : Dictionary containing team statistics
    """
    # Load historical data
    df = pd.read_csv(data_file)
    
    # Filter matches where this team played (either home or away)
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    # Sort by date to get most recent matches
    team_matches['Date'] = pd.to_datetime(team_matches['Date'], format='%d/%m/%Y')
    team_matches = team_matches.sort_values('Date', ascending=False)
    
    # Get the last N games
    recent_matches = team_matches.head(num_games)
    
    if len(recent_matches) < num_games:
        print(f"Warning: Only {len(recent_matches)} matches found for {team_name} (requested {num_games})")
    
    # Initialize stats
    stats = {
        'GoalsScored': 0,
        'GoalsConceded': 0,
        'GoalDiff': 0,
        'Points': 0,
        'WinRate': 0,
        'ShotsOnTarget': 0,
        'ShotsOnTargetAgainst': 0,
        'CleanSheetRate': 0,
        'DrawRate': 0,
        'LowScoringRate': 0,
        'FailedToScoreRate': 0,
        'ConversionRate': 0
    }
    
    if len(recent_matches) == 0:
        return stats
    
    # Calculate statistics
    goals_scored = []
    goals_conceded = []
    shots_on_target = []
    shots_on_target_against = []
    points = []
    clean_sheets = 0
    draws = 0
    low_scoring = 0
    failed_to_score = 0
    
    for _, match in recent_matches.iterrows():
        is_home = match['HomeTeam'] == team_name
        
        if is_home:
            gs = match['FTHG']  # Full Time Home Goals
            gc = match['FTAG']  # Full Time Away Goals
            sot = match['HST']  # Home Shots on Target
            sota = match['AST']  # Away Shots on Target
        else:
            gs = match['FTAG']
            gc = match['FTHG']
            sot = match['AST']
            sota = match['HST']
        
        goals_scored.append(gs)
        goals_conceded.append(gc)
        shots_on_target.append(sot)
        shots_on_target_against.append(sota)
        
        # Calculate points
        if gs > gc:
            points.append(3)
        elif gs == gc:
            points.append(1)
            draws += 1
        else:
            points.append(0)
        
        # Clean sheet
        if gc == 0:
            clean_sheets += 1
        
        # Low scoring (total goals <= 1)
        if gs + gc <= 1:
            low_scoring += 1
        
        # Failed to score
        if gs == 0:
            failed_to_score += 1
    
    # Calculate averages
    n = len(recent_matches)
    stats['GoalsScored'] = sum(goals_scored) / n
    stats['GoalsConceded'] = sum(goals_conceded) / n
    stats['GoalDiff'] = stats['GoalsScored'] - stats['GoalsConceded']
    stats['Points'] = sum(points) / n
    stats['WinRate'] = sum([1 for p in points if p == 3]) / n
    stats['ShotsOnTarget'] = sum(shots_on_target) / n
    stats['ShotsOnTargetAgainst'] = sum(shots_on_target_against) / n
    stats['CleanSheetRate'] = clean_sheets / n
    stats['DrawRate'] = draws / n
    stats['LowScoringRate'] = low_scoring / n
    stats['FailedToScoreRate'] = failed_to_score / n
    
    # Conversion rate (goals per shot on target)
    total_shots = sum(shots_on_target)
    stats['ConversionRate'] = sum(goals_scored) / total_shots if total_shots > 0 else 0
    
    return stats

def get_betting_odds(home_team, away_team, api_key=None):
    """
    Fetch current betting odds from The Odds API for a Premier League match.
    
    Parameters:
    -----------
    home_team : str
        Name of home team (e.g., 'Arsenal')
    away_team : str
        Name of away team (e.g., 'Liverpool')
    api_key : str, optional
        Your Odds API key. If None, reads from 'odds_api_key.txt'
    
    Returns:
    --------
    dict : Betting odds in format {'AvgH': float, 'AvgD': float, 'AvgA': float, 
                                    'Avg>2.5': float, 'Avg<2.5': float}
    """
    # Load API key
    if api_key is None:
        try:
            with open('odds_api_key.txt', 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError("API key not found. Create 'odds_api_key.txt' or pass api_key parameter")
    
    # Team name mapping (your names -> Odds API names)
    team_mapping = {
        'Arsenal': 'Arsenal',
        'Aston Villa': 'Aston Villa',
        'Bournemouth': 'AFC Bournemouth',
        'Brentford': 'Brentford',
        'Brighton': 'Brighton and Hove Albion',
        'Burnley': 'Burnley',
        'Chelsea': 'Chelsea',
        'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton',
        'Fulham': 'Fulham',
        'Liverpool': 'Liverpool',
        'Luton': 'Luton Town',
        'Man City': 'Manchester City',
        'Man United': 'Manchester United',
        'Newcastle': 'Newcastle United',
        'Nott\'m Forest': 'Nottingham Forest',
        'Sheffield United': 'Sheffield United',
        'Tottenham': 'Tottenham Hotspur',
        'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton Wanderers',
        'Leicester': 'Leicester City',
        'Leeds': 'Leeds United',
        'Southampton': 'Southampton',
        'Watford': 'Watford',
        'Norwich': 'Norwich City'
    }
    
    # Map team names
    home_api = team_mapping.get(home_team, home_team)
    away_api = team_mapping.get(away_team, away_team)
    
    # API endpoint
    sport = 'soccer_epl'  # English Premier League
    regions = 'uk,eu'  # UK and European bookmakers
    markets = 'h2h,totals'  # Head-to-head (match winner) and totals (over/under)
    odds_format = 'decimal'
    
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/'
    
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format
    }
    
    print(f"\nFetching odds for {home_team} vs {away_team}...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        matches = response.json()
        
        # Find the specific match
        target_match = None
        for match in matches:
            teams = {match['home_team'], match['away_team']}
            if home_api in teams and away_api in teams:
                # Verify home/away alignment
                if match['home_team'] == home_api:
                    target_match = match
                    break
        
        if not target_match:
            print(f"No live odds found for {home_team} vs {away_team}")
            print(f"\nAvailable upcoming matches:")
            for i, match in enumerate(matches[:10], 1):  # Show first 10 matches
                home = match['home_team']
                away = match['away_team']
                time = match['commence_time']
                print(f"   {i}. {home} vs {away} ({time})")
            print(f"\n   Using default odds based on typical Premier League match")
            return {
                'AvgH': 2.50,
                'AvgD': 3.40,
                'AvgA': 2.80,
                'Avg>2.5': 1.80,
                'Avg<2.5': 2.05
            }
        
        # Extract odds from all bookmakers
        home_odds = []
        draw_odds = []
        away_odds = []
        over_25_odds = []
        under_25_odds = []
        
        for bookmaker in target_match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    # Head-to-head market (match winner)
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_api:
                            home_odds.append(outcome['price'])
                        elif outcome['name'] == away_api:
                            away_odds.append(outcome['price'])
                        elif outcome['name'] == 'Draw':
                            draw_odds.append(outcome['price'])
                
                elif market['key'] == 'totals':
                    # Totals market (over/under 2.5 goals)
                    for outcome in market['outcomes']:
                        if outcome['name'] == 'Over' and outcome['point'] == 2.5:
                            over_25_odds.append(outcome['price'])
                        elif outcome['name'] == 'Under' and outcome['point'] == 2.5:
                            under_25_odds.append(outcome['price'])
        
        # Calculate averages
        odds = {
            'AvgH': np.mean(home_odds) if home_odds else 2.50,
            'AvgD': np.mean(draw_odds) if draw_odds else 3.40,
            'AvgA': np.mean(away_odds) if away_odds else 2.80,
            'Avg>2.5': np.mean(over_25_odds) if over_25_odds else 1.80,
            'Avg<2.5': np.mean(under_25_odds) if under_25_odds else 2.05
        }
        
        print(f"✓ Fetched odds from {len(target_match['bookmakers'])} bookmakers")
        print(f"  Match time: {target_match.get('commence_time', 'N/A')}")
        
        return odds
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(f"   Using default odds")
        return {
            'AvgH': 2.50,
            'AvgD': 3.40,
            'AvgA': 2.80,
            'Avg>2.5': 1.80,
            'Avg<2.5': 2.05
        }


def predict_match(home_team, away_team, 
                  betting_odds=None,
                  home_stats_L5=None, away_stats_L5=None,
                  home_stats_L10=None, away_stats_L10=None,
                  data_file='data_with_features_complete.csv'):
    """
    Predict the outcome of a Premier League match.
    
    Parameters:
    -----------
    home_team : str
        Name of home team (e.g., 'Arsenal')
    away_team : str
        Name of away team (e.g., 'Liverpool')
    betting_odds : dict, optional
        Current betting odds: {'AvgH': 2.10, 'AvgD': 3.40, 'AvgA': 3.20, 
                              'Avg>2.5': 1.80, 'Avg<2.5': 2.10}
        If None, will fetch live odds from API
    home_stats_L5 : dict, optional
        Home team's last 5 games stats. If None, will auto-calculate from data_file
    away_stats_L5 : dict, optional
        Away team's last 5 games stats. If None, will auto-calculate from data_file
    home_stats_L10 : dict, optional
        Home team's last 10 games stats. If None, will auto-calculate from data_file
    away_stats_L10 : dict, optional
        Away team's last 10 games stats. If None, will auto-calculate from data_file
    data_file : str, optional
        Path to CSV file with historical data (default: 'data_with_features_complete.csv')
    
    Returns:
    --------
    dict : Prediction results with predicted outcome and probabilities
    """
    
    # Fetch betting odds if not provided
    if betting_odds is None:
        betting_odds = get_betting_odds(home_team, away_team)
    
    # Auto-calculate team stats if not provided
    if home_stats_L5 is None:
        print(f"Calculating {home_team} statistics from recent matches...")
        home_stats_L5 = get_team_recent_stats(home_team, data_file, num_games=5)
    
    if away_stats_L5 is None:
        print(f"Calculating {away_team} statistics from recent matches...")
        away_stats_L5 = get_team_recent_stats(away_team, data_file, num_games=5)
    
    if home_stats_L10 is None:
        home_stats_L10 = get_team_recent_stats(home_team, data_file, num_games=10)
    
    if away_stats_L10 is None:
        away_stats_L10 = get_team_recent_stats(away_team, data_file, num_games=10)
    
    # Load the trained model
    print("\nLoading trained model...")
    with open('premier_league_model.pkl', 'rb') as f:       # 'rb' mode (read binary) as pickle files are binary, not text 
        model = pickle.load(f)      # Deserialises the RandomForestClassifier that was trained and saved in train_model.py
    
    # Load feature names (must match training exactly!)
    feature_names = pd.read_csv('model_features.csv')['FeatureName'].tolist()
    
    print(f"\n{'='*60}")
    print(f"PREDICTING: {home_team} vs {away_team}")
    print(f"{'='*60}")
    
    # Create feature dictionary
    features = {}
    
    # Add Home team L5 stats
    for stat_name, value in home_stats_L5.items():
        features[f'Home_{stat_name}_L5'] = value
    
    # Add Away team L5 stats
    for stat_name, value in away_stats_L5.items():
        features[f'Away_{stat_name}_L5'] = value
    
    # Add Home team L10 stats
    for stat_name, value in home_stats_L10.items():
        features[f'Home_{stat_name}_L10'] = value
    
    # Add Away team L10 stats
    for stat_name, value in away_stats_L10.items():
        features[f'Away_{stat_name}_L10'] = value
    
    # Calculate gap features (strength differences)
    for window in [5, 10]:
        home_gd = home_stats_L10['GoalDiff'] if window == 10 else home_stats_L5['GoalDiff']
        away_gd = away_stats_L10['GoalDiff'] if window == 10 else away_stats_L5['GoalDiff']
        features[f'GoalDiff_Gap_L{window}'] = abs(home_gd - away_gd)
        
        home_pts = home_stats_L10['Points'] if window == 10 else home_stats_L5['Points']
        away_pts = away_stats_L10['Points'] if window == 10 else away_stats_L5['Points']
        features[f'Points_Gap_L{window}'] = abs(home_pts - away_pts)
        
        home_shots = home_stats_L10['ShotsOnTarget'] if window == 10 else home_stats_L5['ShotsOnTarget']
        away_shots = away_stats_L10['ShotsOnTarget'] if window == 10 else away_stats_L5['ShotsOnTarget']
        features[f'ShotsOnTarget_Gap_L{window}'] = abs(home_shots - away_shots)
        
        home_conv = home_stats_L10['ConversionRate'] if window == 10 else home_stats_L5['ConversionRate']
        away_conv = away_stats_L10['ConversionRate'] if window == 10 else away_stats_L5['ConversionRate']
        features[f'ConversionRate_Gap_L{window}'] = abs(home_conv - away_conv)
    
    # Calculate combined features (both teams together)
    for window in [5, 10]:
        home_cs = home_stats_L10['CleanSheetRate'] if window == 10 else home_stats_L5['CleanSheetRate']
        away_cs = away_stats_L10['CleanSheetRate'] if window == 10 else away_stats_L5['CleanSheetRate']
        features[f'Combined_CleanSheetRate_L{window}'] = (home_cs + away_cs) / 2
        
        home_dr = home_stats_L10['DrawRate'] if window == 10 else home_stats_L5['DrawRate']
        away_dr = away_stats_L10['DrawRate'] if window == 10 else away_stats_L5['DrawRate']
        features[f'Combined_DrawRate_L{window}'] = (home_dr + away_dr) / 2
        
        home_ls = home_stats_L10['LowScoringRate'] if window == 10 else home_stats_L5['LowScoringRate']
        away_ls = away_stats_L10['LowScoringRate'] if window == 10 else away_stats_L5['LowScoringRate']
        features[f'Combined_LowScoringRate_L{window}'] = (home_ls + away_ls) / 2
        
        home_fts = home_stats_L10['FailedToScoreRate'] if window == 10 else home_stats_L5['FailedToScoreRate']
        away_fts = away_stats_L10['FailedToScoreRate'] if window == 10 else away_stats_L5['FailedToScoreRate']
        features[f'Combined_FailedToScoreRate_L{window}'] = (home_fts + away_fts) / 2
    
    # Add betting odds features
    features['Prob_HomeWin'] = 1 / betting_odds['AvgH']
    features['Prob_Draw'] = 1 / betting_odds['AvgD']
    features['Prob_AwayWin'] = 1 / betting_odds['AvgA']
    features['Prob_Over2.5Goals'] = 1 / betting_odds['Avg>2.5']
    features['Prob_Under2.5Goals'] = 1 / betting_odds['Avg<2.5']
    
    # Create DataFrame with features in correct order
    X = pd.DataFrame([features], columns=feature_names)     # [features] wraps the single dictionary into a list containing one dictionary 
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map prediction to readable outcome
    outcome_map = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}
    
    print(f"\nINPUT FEATURES:")
    print(f"   Bookmaker Odds:")
    print(f"     Home Win: {betting_odds['AvgH']:.2f} → {features['Prob_HomeWin']:.1%} probability")
    print(f"     Draw:     {betting_odds['AvgD']:.2f} → {features['Prob_Draw']:.1%} probability")
    print(f"     Away Win: {betting_odds['AvgA']:.2f} → {features['Prob_AwayWin']:.1%} probability")
    
    """
    print(f"\n   Recent Form (Last 10 games):")
    print(f"     {home_team}: Goal Diff {home_stats_L10['GoalDiff']:.1f}, Points {home_stats_L10['Points']:.1f}/game")
    print(f"     {away_team}: Goal Diff {away_stats_L10['GoalDiff']:.1f}, Points {away_stats_L10['Points']:.1f}/game")
    """
    
    print(f"\nMODEL PREDICTION:")
    print(f"   Predicted Outcome: {outcome_map[prediction]}")
    print(f"\n   Confidence Levels:")
    print(f"     {home_team} Win: {probabilities[2]:.1%}")
    print(f"     Draw: {probabilities[1]:.1%}")
    print(f"     {away_team} Win: {probabilities[0]:.1%}")
    
    print(f"\n{'='*60}\n")
    
    return {
        'prediction': outcome_map[prediction],
        'probabilities': {
            'home_win': probabilities[2],
            'draw': probabilities[1],
            'away_win': probabilities[0]
        }
    }


# EXAMPLE USAGE
if __name__ == "__main__":
    # Example 1: Automatic stats extraction AND odds fetching (RECOMMENDED)
    # Just provide team names - everything else is fetched automatically!
    
    home_team = 'Chelsea'
    away_team = 'Brentford'
    
    print("=" * 70)
    print(f"FULL AUTOMATIC PREDICTION - {home_team} vs {away_team}")
    print("=" * 70)
    
    result = predict_match(
        home_team,
        away_team
        # No betting_odds needed - fetched automatically from Odds API!
        # No stats needed - calculated automatically from historical data!
    )
    
    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)
    
    # Example 2: Hardcoded stats (if you want to override)
    """
    home_stats_L5 = {
        'GoalsScored': 2.2,
        'GoalsConceded': 0.8,
        'GoalDiff': 1.4,
        'Points': 2.4,
        'WinRate': 0.8,
        'ShotsOnTarget': 6.2,
        'ShotsOnTargetAgainst': 3.0,
        'CleanSheetRate': 0.6,
        'DrawRate': 0.0,
        'LowScoringRate': 0.2,
        'FailedToScoreRate': 0.0,
        'ConversionRate': 0.35
    }
    
    away_stats_L5 = {
        'GoalsScored': 2.0,
        'GoalsConceded': 1.2,
        'GoalDiff': 0.8,
        'Points': 2.0,
        'WinRate': 0.6,
        'ShotsOnTarget': 5.8,
        'ShotsOnTargetAgainst': 3.5,
        'CleanSheetRate': 0.4,
        'DrawRate': 0.2,
        'LowScoringRate': 0.4,
        'FailedToScoreRate': 0.2,
        'ConversionRate': 0.34
    }
    
    result = predict_match(
        home_team='Arsenal',
        away_team='Liverpool',
        betting_odds=betting_odds,
        home_stats_L5=home_stats_L5,
        away_stats_L5=away_stats_L5
        # home_stats_L10 and away_stats_L10 can also be provided
    )
    """

