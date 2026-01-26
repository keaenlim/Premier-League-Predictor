# Premier League Match Predictor

A machine learning system that predicts English Premier League match outcomes (Win/Draw/Loss) using historical data and real-time betting odds.

## Features

- **53%+ prediction accuracy** using Random Forest classifier
- **60+ engineered features** including rolling statistics, form indicators, and betting odds
- **Automated data pipeline** for weekly updates from football-data.co.uk
- **Live betting odds integration** via The Odds API
- **Chronological validation** to prevent data leakage

## Project Structure

```
├── feature_engineering.py      # Create rolling window features (L5/L10)
├── preprocess_data.py          # Clean data, assign team IDs
├── train_model.py              # Train Random Forest model
├── predict_match.py            # Make predictions for upcoming matches
├── update_data.py              # Automated weekly data update script
├── merge_csv.py                # Combine multiple season CSVs
├── show_teams.py               # Display team mappings
└── PREDICTION_GUIDE.md         # Detailed usage guide
```

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn requests
```

### 2. Get API Key (Optional - for live odds)

1. Sign up at [The Odds API](https://the-odds-api.com/)
2. Get your API key
3. Create `odds_api_key.txt` in the project root:
```
your_api_key_here
```

### 3. Get Historical Data

Download Premier League data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php)

- Save CSV files to project directory
- Run `python3 merge_csv.py` to combine seasons

## Usage

### Initial Setup (First Time)

```bash
# 1. Preprocess raw data
python3 preprocess_data.py

# 2. Engineer features
python3 feature_engineering.py

# 3. Train model
python3 train_model.py
```

### Weekly Updates

```bash
# Automatically downloads latest data and retrains
python3 update_data.py
```

### Make Predictions

```bash
# Method 1: Automatic (fetches live odds + calculates stats)
python3 predict_match.py
```

Edit `predict_match.py` to change teams:
```python
home_team = 'Arsenal'
away_team = 'Liverpool'
result = predict_match(home_team, away_team)
```

## Model Details

- **Algorithm**: Random Forest (100 estimators)
- **Features**: 60+ including:
  - Rolling averages (goals, shots, clean sheets) over L5/L10 games
  - Team form indicators (win rate, draw rate, points)
  - Shot conversion rates
  - Matchup comparison metrics (goal diff gap, points gap)
  - Live betting odds (if available)
- **Validation**: Time-based train/test split (80/20)
- **Accuracy**: 53%+ on test set

## Feature Engineering

Key features prevent data leakage by only using historical data:

- **Rolling Windows**: Last 5 and 10 games statistics
- **Form Metrics**: Goals scored/conceded, points, win rate
- **Defensive Strength**: Clean sheet rates, goals conceded
- **Attacking Quality**: Shots on target, conversion rates
- **Matchup Gaps**: Difference in form between teams
- **Betting Market**: Live odds as feature (optional)

## Data Sources

- **Historical Matches**: [football-data.co.uk](https://www.football-data.co.uk/)
- **Live Odds**: [The Odds API](https://the-odds-api.com/)

## Files Excluded from Git

- `odds_api_key.txt` - API key (sensitive)
- `*.csv` - Large data files (regenerate with update_data.py)
- `*.pkl` - Trained models (regenerate with train_model.py)
- `__pycache__/` - Python cache

## Next Steps

- [ ] Add more features (injuries, weather, head-to-head history)
- [ ] Experiment with other algorithms (XGBoost, Neural Networks)
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Build web interface for predictions
- [ ] Add backtesting framework

## License

MIT License - feel free to use and modify

## Author

Built as a machine learning portfolio project
