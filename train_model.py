import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_prediction_model(input_file='data_with_features_complete.csv', test_size=0.2):
    """
    Train a Random Forest model to predict Premier League match outcomes.
    
    Steps:
    1. Load data with engineered features
    2. Select feature columns and target variable
    3. Split data chronologically (time-based split)
    4. Train Random Forest classifier
    5. Evaluate performance
    6. Analyze feature importance
    
    Parameters:
    input_file : str
        Path to CSV with complete features
    test_size : float
        Proportion of most recent data to use for testing (0.2 = 20%)
    """
    
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(input_file)
    
    # Convert date for potential time-based operations
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    print(f"Loaded {len(df)} matches with complete features")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\n" + "="*60)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("="*60)
    
    # Target variable: FTR (Full Time Result)
    # H = Home Win, D = Draw, A = Away Win
    target = df['FTR'].copy()
    print(f"\nTarget variable: FTR (Full Time Result)")
    print(f"  Value counts:")
    print(target.value_counts().to_string())
    
    # STEP 2A: Convert betting odds to probabilities
    # Odds tell us how much money you'd win, but probabilities are easier to understand
    # Formula: Probability = 1 / Odds
    # Example: Odds of 2.0 means 50% probability (1/2.0 = 0.5)
    
    print(f"\nConverting betting odds to probabilities...")
    
    # Check if betting odds exist in the data
    if 'AvgH' in df.columns and 'AvgD' in df.columns and 'AvgA' in df.columns:
        # 1. Match Result Probabilities
        df['Prob_HomeWin'] = 1 / df['AvgH']  # Higher odds = lower probability
        df['Prob_Draw'] = 1 / df['AvgD']
        df['Prob_AwayWin'] = 1 / df['AvgA']
        
        print(f"✓ Match Result Odds: 3 features")
        print(f"  - Prob_HomeWin, Prob_Draw, Prob_AwayWin")
        
        # 2. Over/Under 2.5 Goals Probabilities (IMPORTANT for draw prediction!)
        # Low-scoring games (under 2.5) often end in draws
        if 'Avg>2.5' in df.columns and 'Avg<2.5' in df.columns:
            df['Prob_Over2.5Goals'] = 1 / df['Avg>2.5']  # Probability of 3+ goals
            df['Prob_Under2.5Goals'] = 1 / df['Avg<2.5']  # Probability of 0-2 goals
            
            print(f"✓ Over/Under 2.5 Goals Odds: 2 features")
            print(f"  - Prob_Over2.5Goals (high-scoring match expected)")
            print(f"  - Prob_Under2.5Goals (low-scoring match expected → draws!)")
            ou_available = True
        else:
            ou_available = False
        
        # 3. Asian Handicap (shows strength difference)
        # Small handicap = evenly matched = draw likely
        if 'AvgAHH' in df.columns and 'AvgAHA' in df.columns:
            df['AsianHandicap_Home'] = df['AvgAHH']
            df['AsianHandicap_Away'] = df['AvgAHA']
            # Calculate the implied handicap line
            # This shows how many goals bookmakers think the favorite is better by
            
            print(f"✓ Asian Handicap Odds: 2 features")
            print(f"  - AsianHandicap_Home")
            print(f"  - AsianHandicap_Away")
            print(f"  (Small values = evenly matched teams = draw likely)")
            ah_available = True
        else:
            ah_available = False
        
        # Example to show what we did
        print(f"\nExample conversion (first match):")
        print(f"  Match Result:")
        print(f"    AvgH odds: {df['AvgH'].iloc[0]:.2f} → Prob_HomeWin: {df['Prob_HomeWin'].iloc[0]:.1%}")
        print(f"    AvgD odds: {df['AvgD'].iloc[0]:.2f} → Prob_Draw: {df['Prob_Draw'].iloc[0]:.1%}")
        print(f"    AvgA odds: {df['AvgA'].iloc[0]:.2f} → Prob_AwayWin: {df['Prob_AwayWin'].iloc[0]:.1%}")
        
        if ou_available:
            print(f"  Goals Expected:")
            print(f"    Avg>2.5: {df['Avg>2.5'].iloc[0]:.2f} → Prob_Over2.5: {df['Prob_Over2.5Goals'].iloc[0]:.1%}")
            print(f"    Avg<2.5: {df['Avg<2.5'].iloc[0]:.2f} → Prob_Under2.5: {df['Prob_Under2.5Goals'].iloc[0]:.1%}")
        
        betting_odds_available = True
    else:
        print(f"⚠️  Betting odds not found in dataset")
        betting_odds_available = False
        ou_available = False
        ah_available = False
    
    # Feature columns: All the rolling average features we created
    # Pattern: Home_* and Away_* features with L5 and L10 windows
    feature_cols = [col for col in df.columns if 
                   (col.startswith('Home_') or col.startswith('Away_')) and 
                   ('_L5' in col or '_L10' in col)]
    
    # Add betting odds probabilities to features (if available)
    odds_count = 0
    if betting_odds_available:
        odds_features = ['Prob_HomeWin', 'Prob_Draw', 'Prob_AwayWin']
        feature_cols.extend(odds_features)
        odds_count += 3
        
        if ou_available:
            ou_features = ['Prob_Over2.5Goals', 'Prob_Under2.5Goals']
            feature_cols.extend(ou_features)
            odds_count += 2
        
        if ah_available:
            ah_features = ['AsianHandicap_Home', 'AsianHandicap_Away']
            feature_cols.extend(ah_features)
            odds_count += 2
        
        print(f"\n✓ Added {odds_count} betting odds features to the model")
    
    print(f"\n{len(feature_cols)} total features selected:")
    print(f"  - {len(feature_cols) - odds_count} statistical features")
    if odds_count > 0:
        print(f"  - {odds_count} betting odds features")
    print(f"  Examples: {feature_cols[:5]}")
    print(f"  ... (showing first 5 of {len(feature_cols)})")
    
    # Create feature matrix X and target vector y
    X = df[feature_cols].copy()
    y = target.copy()
    
    # Check for any remaining NaN values (there shouldn't be any)
    if X.isnull().sum().sum() > 0:
        print(f"\n⚠️  Warning: Found {X.isnull().sum().sum()} NaN values in features")
        print("Dropping rows with NaN...")
        valid_idx = X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
    
    print(f"\nFinal dataset: {len(X)} matches, {len(feature_cols)} features")
    
    print("\n" + "="*60)
    print("STEP 3: TRAIN-TEST SPLIT (TIME-BASED)")
    print("="*60)
    
    # TIME-BASED SPLIT (not random!)
    # Use the most recent 20% of matches for testing
    # This simulates real prediction: train on past, predict future
    split_idx = int(len(df) * (1 - test_size))

    # X represents features (32 rolling averages), y represents the match results (H, D, A) 
    # Each row represents a match, while each column is the data that corresponds to that specific field
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Get the date split point
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    split_date = df_sorted.iloc[split_idx]['Date']
    
    print(f"Train set: {len(X_train)} matches (oldest matches)")
    print(f"Test set:  {len(X_test)} matches (most recent matches)")
    print(f"Split date: {split_date.strftime('%Y-%m-%d')}")
    print(f"\nWhy time-based? In real life, we use past data to predict future matches!")
    
    print("\n" + "="*60)
    print("STEP 4: TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # Random Forest Classifier
    # - Ensemble of decision trees (100 trees by default)
    # - Each tree votes on the outcome, majority wins
    # - Good for: tabular data, handles non-linear relationships, resistant to overfitting
    # - random_state=42 ensures reproducible results
    
    print("\nInitializing Random Forest Classifier...")
    print("  - n_estimators=100 (100 decision trees)")
    print("  - NO depth constraints (let the model learn freely)")
    print("  - random_state=42 (for reproducibility)")
    print("  - class_weight='balanced' (handle uneven H/D/A distribution)")
    
    # NO CONSTRAINTS! Let the model use all the features freely
    # Yes, training accuracy will be 100% (overfitting)
    # But test accuracy is what matters, and it's BETTER this way!
    model = RandomForestClassifier(
        n_estimators=100,           # Number of decision trees in the forest
        # max_depth=8,              # Limit tree depth to prevent overfitting
        # min_samples_leaf=20,      # Require minimum samples per leaf node
        # min_samples_split=40,     # Require minimum samples to split a node
        # max_features='sqrt',      # Limit features considered per tree
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    # Make predictions on both train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nAccuracy Scores:")
    print(f"  Training set: {train_accuracy:.1%}")
    print(f"  Test set:     {test_accuracy:.1%}")
    
    if train_accuracy - test_accuracy > 0.15:
        print(f"\n Large gap suggests overfitting (memorizing training data)")
    else:
        print(f"\n✓ Reasonable train-test gap")
    
    # Detailed classification report for test set
    print(f"\nDetailed Performance on Test Set:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Away Win', 'Draw', 'Home Win']))
    
    # Confusion matrix
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred, labels=['A', 'D', 'H'])
    
    # Create a nice formatted confusion matrix
    cm_df = pd.DataFrame(cm, 
                         index=['Actual: Away Win', 'Actual: Draw', 'Actual: Home Win'],
                         columns=['Pred: Away Win', 'Pred: Draw', 'Pred: Home Win'])
    print(cm_df.to_string())
    
    print("\nHow to read confusion matrix:")
    print("  - Diagonal = Correct predictions")
    print("  - Off-diagonal = Mistakes")
    print("  - Example: If [Actual: Draw, Pred: Home Win] = 50, we predicted")
    print("    Home Win 50 times when the actual result was a Draw")
    
    print("\n" + "="*60)
    print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Feature importance shows which features the model relies on most
    # Higher importance = more useful for predictions
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"\n✓ Full feature importance saved to feature_importance.csv")
    
    print("\n" + "="*60)
    print("STEP 7: SAVE MODEL")
    print("="*60)
    
    # Save the trained model for future use
    import pickle
    
    model_filename = 'premier_league_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to {model_filename}")
    print(f"\nYou can load it later with:")
    print(f"  with open('{model_filename}', 'rb') as f:")
    print(f"      model = pickle.load(f)")
    
    # Save feature column names (needed when making predictions)
    feature_cols_df = pd.DataFrame({'FeatureName': feature_cols})
    feature_cols_df.to_csv('model_features.csv', index=False)
    print(f"✓ Feature names saved to model_features.csv")
    
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"✓ Model Type: Random Forest (100 trees)")
    print(f"✓ Training Matches: {len(X_train)}")
    print(f"✓ Test Matches: {len(X_test)}")
    print(f"✓ Test Accuracy: {test_accuracy:.1%}")
    print(f"✓ Features Used: {len(feature_cols)}")
    print(f"✓ Most Important Feature: {feature_importance.iloc[0]['Feature']}")
    
    return model, feature_importance, X_test, y_test, y_test_pred

if __name__ == "__main__":
    print("\nPREMIER LEAGUE MATCH OUTCOME PREDICTOR\n")
    
    # Train the model
    model, feature_imp, X_test, y_test, y_pred = train_prediction_model(
        input_file='data_with_features_complete.csv',
        test_size=0.2  # Use last 20% of matches for testing
    )

    '''
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review feature importance - which stats matter most?")
    print("2. Analyze confusion matrix - what mistakes does the model make?")
    print("3. Try improving the model:")
    print("   - Add more features (head-to-head, betting odds, etc.)")
    print("   - Tune hyperparameters (tree depth, number of trees)")
    print("   - Try different models (XGBoost, Neural Networks)")
    print("4. Use the model to predict upcoming matches!")
    print("="*60)
    '''
