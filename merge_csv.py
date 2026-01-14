import pandas as pd
import glob
from datetime import datetime

# Get all CSV files in the current directory
csv_files = glob.glob('*.csv')

# Filter out the output file if it already exists
csv_files = [f for f in csv_files if f != 'merged_data.csv']

print(f"Found {len(csv_files)} CSV files: {csv_files}")

# Read and combine all CSV files
dfs = []
for file in csv_files:
    print(f"Reading {file}...")
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows before sorting: {len(merged_df)}")

# Convert Date column to datetime for proper sorting
# The date format appears to be DD/MM/YYYY
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d/%m/%Y')

# Sort by date
merged_df = merged_df.sort_values('Date')

# Convert date back to original format for output
merged_df['Date'] = merged_df['Date'].dt.strftime('%d/%m/%Y')

# Save the merged and sorted data
output_file = 'merged_data.csv'
merged_df.to_csv(output_file, index=False)

print(f"\nMerged data saved to {output_file}")
print(f"Total rows: {len(merged_df)}")
print(f"Date range: {merged_df['Date'].iloc[0]} to {merged_df['Date'].iloc[-1]}")
