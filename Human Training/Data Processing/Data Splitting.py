import os
import csv

def split_and_insert_empty_rows(input_file, chunk_size=16):
    input_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_dir, "split_data.csv")

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        chunk = []
        for row_number, row in enumerate(reader, start=1):
            chunk.append(row)
            if row_number % chunk_size == 0:
                writer.writerows(chunk)
                writer.writerow([])  # Insert empty row
                chunk = []

        # Handle the last chunk if it's not empty
        if chunk:
            writer.writerows(chunk)

# Example usage
input_file = "R:\\OHG\\Cardiac_Respiratory_Phantom\\Blood Pressure Processing\\Training\\Human Training\\Data Management\\HugoHR.CSV"
split_and_insert_empty_rows(input_file, chunk_size=16)