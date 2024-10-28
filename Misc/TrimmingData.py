import pandas as pd

# Load the CSV file
data = pd.read_csv('data.csv')

# Assuming the second column is at index 1 and has data formatted as [BPM,RR,PSI]
data[['BPM', 'RR', 'PSI']] = data.iloc[:, 1].str.strip('[]').str.split(',', expand=True)

# Convert BPM and RR to numeric
data['BPM'] = pd.to_numeric(data['BPM'], errors='coerce')
data['RR'] = pd.to_numeric(data['RR'], errors='coerce')

# Filter the data based on the specified ranges
filtered_data = data[(data['BPM'].between(80, 100)) & (data['RR'].between(20, 30))]

# Save the filtered data to a new CSV file
filtered_data.to_csv('TrimmedData.csv', index=False)

print("Filtered data has been saved to TrimmedData.csv")
