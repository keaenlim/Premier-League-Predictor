import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data_with_features_complete.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
target = df['FTR'].copy()

# Create betting odds features
df['Prob_HomeWin'] = 1 / df['AvgH']
df['Prob_Draw'] = 1 / df['AvgD']
df['Prob_AwayWin'] = 1 / df['AvgA']
df['Prob_Over2.5Goals'] = 1 / df['Avg>2.5']
df['Prob_Under2.5Goals'] = 1 / df['Avg<2.5']

# ONLY use betting odds features (no statistical features!)
feature_cols = ['Prob_HomeWin', 'Prob_Draw', 'Prob_AwayWin', 'Prob_Over2.5Goals', 'Prob_Under2.5Goals']

X = df[feature_cols].copy()
y = target.copy()

valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("="*60)
print("TEST 1: BETTING ODDS ONLY (5 features, no constraints)")
print("="*60)

model1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model1.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model1.predict(X_train))
test_acc = accuracy_score(y_test, model1.predict(X_test))

print(f'Training: {train_acc:.1%}')
print(f'Test: {test_acc:.1%}')
print(f'Gap: {train_acc - test_acc:.1%}')

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, model1.predict(X_test), labels=['A', 'D', 'H'])
print(pd.DataFrame(cm, index=['Actual: A', 'Actual: D', 'Actual: H'],
                   columns=['Pred: A', 'Pred: D', 'Pred: H']))

print("\n" + "="*60)
print("TEST 2: ALL FEATURES (55 features, no constraints)")
print("="*60)

# Now test with ALL features
all_feature_cols = [col for col in df.columns if 
                   (col.startswith('Home_') or col.startswith('Away_') or 
                    col.startswith('GoalDiff_Gap') or col.startswith('Points_Gap') or
                    col.startswith('ShotsOnTarget_Gap') or col.startswith('ConversionRate_Gap') or
                    col.startswith('Combined_') or col.startswith('Prob_')) and 
                   ('_L5' in col or '_L10' in col or 'Prob_' in col or 'Gap_L' in col or 'Combined_' in col)]

X_all = df[all_feature_cols].copy()
y_all = target.copy()

valid_idx = X_all.notna().all(axis=1)
X_all = X_all[valid_idx]
y_all = y_all[valid_idx]

split_idx = int(len(X_all) * 0.8)
X_train_all = X_all.iloc[:split_idx]
X_test_all = X_all.iloc[split_idx:]
y_train_all = y_all.iloc[:split_idx]
y_test_all = y_all.iloc[split_idx:]

model2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model2.fit(X_train_all, y_train_all)

train_acc2 = accuracy_score(y_train_all, model2.predict(X_train_all))
test_acc2 = accuracy_score(y_test_all, model2.predict(X_test_all))

print(f'Training: {train_acc2:.1%}')
print(f'Test: {test_acc2:.1%}')
print(f'Gap: {train_acc2 - test_acc2:.1%}')

print("\nConfusion Matrix:")
cm2 = confusion_matrix(y_test_all, model2.predict(X_test_all), labels=['A', 'D', 'H'])
print(pd.DataFrame(cm2, index=['Actual: A', 'Actual: D', 'Actual: H'],
                   columns=['Pred: A', 'Pred: D', 'Pred: H']))
