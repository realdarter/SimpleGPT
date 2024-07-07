import os
import csv

def ensure_file_exists(file_path):
    """
    Ensures the directory for the specified file path exists. 
    If the file does not exist, creates the necessary directories and an empty file.
    Returns:
    - bool: True if the file already exists or was created, False if there was an error.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.isfile(file_path):
        try:
            with open(file_path, 'w', encoding='utf-8'):
                pass
        except IOError:
            print(f"Error: Could not create file {file_path}")
            return False
    
    return True

def read_txt_file(file_path):
    """
    Reads a text file and returns its contents as a list of strings.
    Each line in the text file is considered a separate context.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def encode_csv(csv_path, out_path='csv_encoded.txt', header=True,
               start_token="<|startoftext|>",
               end_token="<|endoftext|>"):
    """ 
    Encodes a CSV with multiple columns to a format suitable for gpt-2-simple.
    Automatically adds the specified prefix and suffix tokens and includes end_token between each extra column.
    Encodes a CSV file `csv_path` to `out_path` with optional `header` and tokens `start_token`, `end_token`.
    """
    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f, \
         open(out_path, 'w', encoding='utf8', errors='ignore') as w:
        if header:
            f.readline()
        for row in csv.reader(f):
            w.write(start_token + " " + " ".join([f"{cell} {end_token}" for cell in row]) + "\n")
    return out_path

# Example usage
if __name__ == "__main__":
    save_path = 'checkpoint\\run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_csv_path = os.path.join(save_path, 'csv_encoded.txt')
    
    if ensure_file_exists(csv_path):
        encode_csv(csv_path, encoded_csv_path, header=True)