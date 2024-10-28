import csv
from tqdm import tqdm
import os

# File paths
input_file = "R:\\OHG\\Cardiac_Respiratory_Phantom\\Blood Pressure Processing\\Training\\Human Training\\Data Management\\HugoHR.CSV"
output_file = os.path.join(os.path.dirname(input_file), "preprocessed_data_HR.csv")

# Function to preprocess radar data by converting hex-like string into byte-level integers
def preprocess_radar_data(radar_string):
    byte_values = []
    for i in range(0, len(radar_string), 2):
        try:
            byte_value = int(radar_string[i:i+2], 16)
            byte_values.append(byte_value)
        except ValueError:
            print(f"Invalid hex value: {radar_string[i:i+2]}")
            return None
    return byte_values

# Reading the original data and creating a new preprocessed file
with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as newfile:
    reader = csv.reader(csvfile)

    # Skip the header row
    next(reader)

    # Initialize a writer object
    writer = csv.writer(newfile)

    for row_number, row in enumerate(tqdm(reader, desc="Processing Radar Data", unit="row"), start=2):
        if not row:  # Check for empty row (block separator)
            continue  # Skip to the next row

        radar_data = row[0]
        labels = row[1]

        preprocessed_data = preprocess_radar_data(radar_data)

        if preprocessed_data is not None:
            writer.writerow(preprocessed_data + [labels])