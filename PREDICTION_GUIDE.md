# Automated Match Prediction - Usage Guide

Your `predict_match.py` script now **automatically extracts team statistics** from historical data! No more hardcoding stats manually.

## How It Works
### (Automated - 3 lines needed!)
```python
result = predict_match(
    home_team='Arsenal',
    away_team='Liverpool',
    betting_odds={'AvgH': 2.10, 'AvgD': 3.40, 'AvgA': 3.50, 
                  'Avg>2.5': 1.65, 'Avg<2.5': 2.25}
)
```

## What Gets Auto-Calculated

The script automatically:
1. Finds the team's last 5 and 10 matches from `data_with_features_complete.csv`
2. Calculates 12 statistics for each time window:
   - Goals scored/conceded per game
   - Points per game
   - Win rate
   - Shots on target
   - Clean sheet rate
   - Draw rate
   - Low-scoring rate
   - Failed to score rate
   - Conversion rate

3. Computes gap features (strength differences between teams)
4. Computes combined features (both teams together)

## Quick Start

### Simple Prediction
```python
from predict_match import predict_match

result = predict_match(
    home_team='Man City',
    away_team='Chelsea',
    betting_odds={
        'AvgH': 1.50,
        'AvgD': 4.20,
        'AvgA': 6.50,
        'Avg>2.5': 1.40,
        'Avg<2.5': 3.00
    }
)

print(f"Prediction: {result['prediction']}")
print(f"Home Win: {result['probabilities']['home_win']:.1%}")
print(f"Draw: {result['probabilities']['draw']:.1%}")
print(f"Away Win: {result['probabilities']['away_win']:.1%}")
```

### Run Examples
```bash
python3 predict_match.py       # Predicts Arsenal vs Liverpool
python3 predict_example.py     # Predicts 3 different matches
```

## 📁 New Functions Added

### `get_team_recent_stats(team_name, data_file, num_games)`
Extracts recent statistics for any team from historical data.

**Parameters:**
- `team_name`: Team name (e.g., 'Arsenal', 'Liverpool')
- `data_file`: CSV file with match history (default: 'data_with_features_complete.csv')
- `num_games`: Number of recent games to analyze (5 or 10)

**Returns:**
Dictionary with 12 calculated statistics

### Updated `predict_match()`
Now has **optional** parameters for stats:
- If you provide stats → uses them directly
- If you don't provide stats → auto-calculates from historical data

## Where to Get Betting Odds

For real predictions, get current odds from:
- **Oddschecker.com** - Aggregates multiple bookmakers
- **Bet365, William Hill, Betfair** - Individual bookmakers
- Average the odds from multiple sources for `AvgH`, `AvgD`, `AvgA`, etc.

## Important Notes

1. **Team Names Must Match** - Use exact names from your CSV:
   - ✅ 'Man City', 'Man United', 'Newcastle'
   - ❌ 'Manchester City', 'MUFC', 'Newcastle United'

2. **Minimum Data Required** - Teams need at least 10 historical matches in your dataset

3. **Recent Data** - Stats are calculated from the team's most recent matches (sorted by date)

4. **Model Accuracy** - The model is ~52% accurate on test data, so treat predictions as informed estimates

## Example Output

```
Calculating Arsenal statistics from recent matches...
Calculating Liverpool statistics from recent matches...

🔄 Loading trained model...

============================================================
PREDICTING: Arsenal vs Liverpool
============================================================

INPUT FEATURES:
   Bookmaker Odds:
     Home Win: 2.10 → 47.6% probability
     Draw:     3.40 → 29.4% probability
     Away Win: 3.50 → 28.6% probability

MODEL PREDICTION:
   Predicted Outcome: Arsenal Win

   Confidence Levels:
     Arsenal Win: 59.0%
     Draw: 13.0%
     Liverpool Win: 28.0%

============================================================
```

## Advanced Usage

If you want to override auto-calculated stats (for what-if scenarios):

```python
result = predict_match(
    home_team='Arsenal',
    away_team='Liverpool',
    betting_odds={...},
    home_stats_L5=custom_home_stats,  # Your custom stats
    # away_stats_L5 will still be auto-calculated
)
```

## Next Steps

You can now easily:
- Predict any upcoming Premier League match
- Test different betting odds scenarios
- Compare model predictions with bookmaker odds
- Build a prediction system for an entire gameweek
