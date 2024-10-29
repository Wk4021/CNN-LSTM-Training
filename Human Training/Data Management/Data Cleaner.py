import os
import csv
from tqdm import tqdm

def remove_empty_rows(input_file):
    input_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_dir, "cleaned_data_HR.csv")

    # Count total rows for progress bar
    with open(input_file, 'r') as infile:
        total_rows = sum(1 for _ in infile)

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        with tqdm(total=total_rows, desc="Processing Rows", unit="row") as pbar:
            for row in reader:
                # Only write the row if it is not empty
                if any(row):  # This checks if the row contains any non-empty values
                    writer.writerow(row)
                pbar.update(1)  # Update the progress bar

# Example usage
input_file = "R:\\OHG\\Cardiac_Respiratory_Phantom\\Blood Pressure Processing\\Training\\Human Training\\Data Management\\preprocessed_data_HR.csv"
remove_empty_rows(input_file)
