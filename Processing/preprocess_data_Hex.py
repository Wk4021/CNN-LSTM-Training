import csv
from tqdm import tqdm  # Import tqdm for progress bar

# File paths
input_file = 'HugoHR.csv'
output_file = 'preprocessed_data_HR.csv'

# Function to preprocess radar data by converting hex-like string into byte-level integers
def preprocess_radar_data(radar_string):
    byte_values = []
    for i in range(0, len(radar_string), 2):
        try:
            # Convert each pair of characters to an integer with base 16
            byte_value = int(radar_string[i:i+2], 16)
            byte_values.append(byte_value)
        except ValueError:
            # If there's an invalid hex string, log it and return an empty list (or handle as you prefer)
            print(f"Invalid hex value: {radar_string[i:i+2]}")
            return None
    return byte_values

# Reading the original data and creating a new preprocessed file
with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as newfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(newfile)

    next(reader)

    # Read all rows first to get the total number of rows for progress bar
    rows = list(reader)
    
    # Use tqdm to create a progress bar
    for row in tqdm(rows, desc="Processing Radar Data", unit="row"):
        radar_data = row[0]  # The 2060-character radar data
        labels = row[1]      # The labels (BPM, RR, PSI)
        
        # Preprocess radar data
        preprocessed_data = preprocess_radar_data(radar_data)
        
        if preprocessed_data is not None:
            # Write preprocessed data and labels into the new CSV file if valid
            writer.writerow(preprocessed_data + [labels])
        else:
            # Log invalid row
            print(f"Skipping invalid row with radar data: {radar_data}")

print("Preprocessing complete! Data saved to preprocessed_data2.csv")
