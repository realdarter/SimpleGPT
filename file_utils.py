import os
import csv

def ensure_file_exists(file_path, create_if_missing=True):
    """
    Ensures the directory for the specified file path exists. 
    If the file does not exist, creates the necessary directories and an empty file if create_if_missing is True.
    Args:
    - file_path (str): The path of the file to check or create.
    - create_if_missing (bool): Whether to create the file if it does not exist. Defaults to True.
    Returns: bool: True if the file already exists or was successfully created, False if there was an error.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.isfile(file_path):
        return True
    
    if create_if_missing:
        try:
            with open(file_path, 'w', encoding='utf-8'):
                pass
            return True
        except IOError:
            print(f"Error: Could not create file {file_path}")
            return False
    else:
        return False

def read_txt_file(file_path):
    """
    Reads a text file and returns its contents as a list of strings.
    Each line in the text file is considered a separate context.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def encode_csv(csv_path, out_path='csv_encoded.txt', header=True, start_token="<|startoftext|>", sep_token = "<|septext|>", end_token="<|endoftext|>"):
    """ 
    Encodes a CSV with multiple columns to a format suitable for GPT-2.
    Automatically adds the specified start, separator, and end tokens.
    Encodes a CSV file `csv_path` to `out_path` with optional `header` and tokens `start_token`, `sep_token`, and `end_token`.
    """
    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f, \
         open(out_path, 'w', encoding='utf8', errors='ignore') as w:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip the header
        for row in reader:
            encoded_row = f"{start_token} " + f" {sep_token} ".join(row) + f" {end_token}\n"
            w.write(encoded_row)
    return out_path

# Example usage
"""
if __name__ == "__main__":
    save_path = 'checkpoint\\run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_csv_path = os.path.join(save_path, 'csv_encoded.txt')
    
    if ensure_file_exists(csv_path):
        encode_csv(csv_path, encoded_csv_path, header=True)

"""