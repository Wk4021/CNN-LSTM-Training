import pandas as pd

# Specify the path to your CSV file
preprocessed_file = 'preprocessed_data.csv'

# Initialize a counter for the number of rows
row_count = 0

# Open the CSV file and read it line by line
with open(preprocessed_file, 'r') as file:
    for line in file:
        row_count += 1

# Print the total number of rows
print(f"Total number of rows in {preprocessed_file}: {row_count}")

# Check if there is sufficient data for processing
if row_count < 2:
    print("Not enough data to proceed.")
else:
    print("Sufficient data available.")
