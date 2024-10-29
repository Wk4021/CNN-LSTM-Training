import os
import csv
from tqdm import tqdm

def split_and_insert_empty_rows(input_file, chunk_size=16):
    input_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_dir, "split_data.csv")

    # Count total rows for progress bar
    with open(input_file, 'r') as infile:
        total_rows = sum(1 for _ in infile)

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        chunk = []
        with tqdm(total=total_rows, desc="Processing Rows", unit="row") as pbar:
            for row in reader:
                chunk.append(row)
                pbar.update(1)  # Update the progress bar
                
                # When the chunk reaches the specified size, write it to the output file
                if len(chunk) == chunk_size:
                    writer.writerows(chunk)
                    writer.writerow([])  # Insert empty row
                    chunk = []

            # Handle the last chunk
            if chunk:  # If there's any remaining rows
                writer.writerows(chunk)  # Write the remaining rows
                # Do not add an empty row after the last chunk

# Example usage
input_file = "R:\\OHG\\Cardiac_Respiratory_Phantom\\Blood Pressure Processing\\Training\\Human Training\\Data Management\\cleaned_data_HR.CSV"
split_and_insert_empty_rows(input_file, chunk_size=16)
