"""
Simple example of how to predict Premier League matches using the trained model.

The script automatically extracts team statistics from historical data,
so you only need to provide:
1. Team names
2. Current betting odds
"""

from predict_match import predict_match

# Example 1: Predict Manchester City vs Chelsea
print("\n" + "="*70)
print("EXAMPLE 1: Manchester City vs Chelsea")
print("="*70 + "\n")

result_1 = predict_match(
    home_team='Man City',
    away_team='Chelsea',
    betting_odds={
        'AvgH': 1.50,  # Man City heavily favored
        'AvgD': 4.20,
        'AvgA': 6.50,
        'Avg>2.5': 1.40,
        'Avg<2.5': 3.00
    }
)

# Example 2: Predict Everton vs Fulham
print("\n" + "="*70)
print("EXAMPLE 2: Everton vs Fulham")
print("="*70 + "\n")

result_2 = predict_match(
    home_team='Everton',
    away_team='Fulham',
    betting_odds={
        'AvgH': 2.20,
        'AvgD': 3.30,
        'AvgA': 3.40,
        'Avg>2.5': 2.00,
        'Avg<2.5': 1.85
    }
)

# Example 3: Predict Tottenham vs Brighton
print("\n" + "="*70)
print("EXAMPLE 3: Tottenham vs Brighton")
print("="*70 + "\n")

result_3 = predict_match(
    home_team='Tottenham',
    away_team='Brighton',
    betting_odds={
        'AvgH': 1.80,
        'AvgD': 3.80,
        'AvgA': 4.20,
        'Avg>2.5': 1.65,
        'Avg<2.5': 2.25
    }
)

print("\n" + "="*70)
print("SUMMARY OF ALL PREDICTIONS")
print("="*70)
print(f"1. Man City vs Chelsea: {result_1['prediction']}")
print(f"   Home: {result_1['probabilities']['home_win']:.1%}, Draw: {result_1['probabilities']['draw']:.1%}, Away: {result_1['probabilities']['away_win']:.1%}")
print(f"\n2. Everton vs Fulham: {result_2['prediction']}")
print(f"   Home: {result_2['probabilities']['home_win']:.1%}, Draw: {result_2['probabilities']['draw']:.1%}, Away: {result_2['probabilities']['away_win']:.1%}")
print(f"\n3. Tottenham vs Brighton: {result_3['prediction']}")
print(f"   Home: {result_3['probabilities']['home_win']:.1%}, Draw: {result_3['probabilities']['draw']:.1%}, Away: {result_3['probabilities']['away_win']:.1%}")
print("="*70 + "\n")
