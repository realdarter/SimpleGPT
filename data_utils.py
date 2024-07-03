import pandas as pd
import logging
from exceptions import ErrorReadingData, TokenizationError
from transformers import GPT2Tokenizer
import os
import io
import codecs

def read_file(file_path):
    """
    Reads a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file to be read.

    Returns:
        pd.DataFrame: DataFrame containing the data read from the CSV file.

    Raises:
        ErrorReadingData: If an error occurs during file reading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")

    # Read file with codecs to handle BOM
    with codecs.open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Use StringIO to create DataFrame
    df = pd.read_csv(io.StringIO(content))

    return df

def preprocess_data(df, model_path='models/124M'):
    """
    Tokenizes the data in the DataFrame using the specified tokenizer model.

    Args:
        df (pd.DataFrame): DataFrame containing 'context' and 'reply' columns.
        model_path (str): Path to the GPT-2 tokenizer model.

    Returns:
        pd.Series: Series containing tokenized data.
    """
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise TokenizationError(f"The specified path '{model_path}' does not exist or is not a directory.")

    # Manually load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, cache_dir=None, revision='main')

    max_length = tokenizer.model_max_length

    def tokenize_function(row):
        try:
            context = str(row['context'])
            reply = str(row['reply'])
            tokens = tokenizer(context + " " + tokenizer.eos_token + " " + reply, return_tensors='pt', truncation=True, max_length=max_length)
            return tokens
        except Exception as e:
            raise TokenizationError(f"Error tokenizing '{context}' + '{reply}': {str(e)}")

    tokenized_data = df.apply(tokenize_function, axis=1)

    for i in range(min(3, len(tokenized_data))):
        logging.info(f"Tokenized example {i}: {tokenized_data.iloc[i]}")

    return tokenized_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    file_path = 'cleaned.csv'

    df = read_file(file_path)
    print("h")
    # Ensure expected column names are present
    expected_columns = ['context', 'reply']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Expected columns '{expected_columns}' not found in the CSV file.")
    print("tokenizing data")
    tokenized_data = preprocess_data(df)

    # Add more code to train the model or other tasks here
