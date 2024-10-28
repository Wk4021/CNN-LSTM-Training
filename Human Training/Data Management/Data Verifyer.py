import csv

def verify_chunks(input_file, chunk_size=16):
    valid_chunks_count = 0
    current_chunk = []
    line_number = 0

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        
        for row in reader:
            line_number += 1
            
            if not row:  # An empty row indicates the end of a chunk
                if len(current_chunk) > 0:
                    if len(current_chunk) == chunk_size:
                        valid_chunks_count += 1
                    else:
                        print(f"Chunk at line {line_number - len(current_chunk) - 1} is not {chunk_size} rows (found {len(current_chunk)})")
                current_chunk = []  # Reset current chunk
            else:
                current_chunk.append(row)

        # Check the last chunk if it wasn't followed by an empty row
        if current_chunk:
            if len(current_chunk) == chunk_size:
                valid_chunks_count += 1
            else:
                print(f"Chunk at line {line_number - len(current_chunk)} is not {chunk_size} rows (found {len(current_chunk)})")

    print(f"Total valid chunks of {chunk_size} rows: {valid_chunks_count}")

# Example usage
output_file = "R:\\OHG\\Cardiac_Respiratory_Phantom\\Blood Pressure Processing\\Training\\Human Training\\Data Management\\split_data.csv"
verify_chunks(output_file, chunk_size=16)
