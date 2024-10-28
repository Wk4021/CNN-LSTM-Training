import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the CSV file
data = pd.read_csv('HugoHR.csv', header=None)

# Extract BPM and RR from the last cell
data[['BPM', 'RR', 'PSI']] = data.iloc[:, -1].apply(ast.literal_eval).apply(pd.Series)

# Convert BPM and RR to numeric for counting
data['BPM'] = pd.to_numeric(data['BPM'], errors='coerce')
data['RR'] = pd.to_numeric(data['RR'], errors='coerce')

# Count occurrences of each combination of BPM and RR
count = data.groupby(['BPM', 'RR']).size().reset_index(name='Count')

# Print combinations with counts less than 600
print("BPM and RR combinations with counts less than 600:")
less_than_600 = count[count['Count'] < 600]
i = 0 
if less_than_600.empty:
    print("No combinations found with counts less than 600.")
else:
    for index, row in less_than_600.iterrows():
        print(f'BPM: {row["BPM"]}, RR: {row["RR"]}, Count: {row["Count"]}')
        i += 1 * 2
        print(i)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(count.apply(lambda x: f'BPM: {x["BPM"]}, RR: {x["RR"]}', axis=1), count['Count'])
plt.xlabel('BPM and RR Combinations')
plt.ylabel('Count')
plt.title('Count of BPM and RR Combinations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()
