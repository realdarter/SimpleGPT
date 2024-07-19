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

def prepare_csv(csv_path, header=True, start_token='', sep_token=''):
    """ 
    Reads a CSV file and returns a list of all items with optional start, separator, and end tokens.
    Args:
        csv_path (str): The path to the CSV file to read.
        header (bool, optional): Whether the CSV file includes a header row. Defaults to True.
        start_token (str, optional): A token to prepend to each row's content. Defaults to ''.
        sep_token (str, optional): A token to insert between items in a row. Defaults to ''.
    Returns: list: A list of formatted strings, each representing a row from the CSV file.
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
    """
    Checks if all necessary files for a GPT-2 model exist in the specified directory.
    Args: model_path (str): The path to the directory containing model files.
    
    Returns: bool: True if all required files exist, False if any file is missing.
    """
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
