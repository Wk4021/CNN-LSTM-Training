import pandas as pd
import ast

# Load the trimmed data CSV file
data = pd.read_csv('preprocessed_data.csv', header=None)

# Extract BPM and RR from the last cell
data[['BPM', 'RR', 'PSI']] = data.iloc[:, -1].apply(ast.literal_eval).apply(pd.Series)

# Initialize DataFrames for storing selected and remaining points
selected_points = pd.DataFrame()
remaining_points = pd.DataFrame()

# Group by BPM and RR
grouped = data.groupby(['BPM', 'RR'])

# Loop through each BPM, RR combination
for (bpm, rr), group in grouped:
    # Randomly select 500 points or all if less than 500
    if len(group) > 500:
        sample = group.sample(n=500, random_state=1)  # Fixed seed for reproducibility
    else:
        sample = group
    
    # Append selected points to selected_points DataFrame
    selected_points = pd.concat([selected_points, sample])
    
    # Append remaining points to remaining_points DataFrame
    remaining = group[~group.index.isin(sample.index)]
    remaining_points = pd.concat([remaining_points, remaining])

# Save selected and remaining points to new CSV files
selected_points.to_csv('TDdata.csv', index=False)
remaining_points.to_csv('Validation.csv', index=False)

print("Selected points saved to TDdata.csv")
print("Remaining points saved to Validation.csv")
