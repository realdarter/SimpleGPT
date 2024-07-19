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

def prepare_csv(csv_path, header=True, start_token="<[BOS]>", sep_token="<[SEP]>"):
    """ 
    Reads a CSV file and returns a list of all items with optional start, separator, and end tokens.
    """
    all_items = []
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        if header:
            next(reader)
        for row in reader:
            stripped_row = [item.strip().replace('"', '') for item in row]
            encoded_row = f"{start_token} " + f" {sep_token} ".join(stripped_row)
            #print(encoded_row)
            all_items.append(encoded_row.strip())
    return all_items

def check_gpt2_models_exist(model_path):
    model_files = [
        'config.json',
        'generation_config.json',
        'merges.txt',
        'model.safetensors',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.json'
    ]
    all_files_exist = True
    for file in model_files:
        file_path = os.path.join(model_path, file)
        absolute_path = os.path.abspath(file_path)  # Get the absolute path
        if not os.path.isfile(absolute_path):
            print(f"File missing: {absolute_path}")
            all_files_exist = False
    return all_files_exist

#items = prepare_csv("checkpoint/run1/cleaned.csv")
#print(items[0])