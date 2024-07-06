import os
import time
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader
import csv

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

def tokenize_text(tokenizer, text, max_length=512):
    """
    Tokenizes a single line of text and ensures it is padded or truncated to a maximum length.
    """
    encoded_text = tokenizer.encode(text.strip(), add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length')
    return encoded_text

def print_special_tokens(tokenizer_name='gpt2'):
    """
    Prints all special tokens recognized by the tokenizer.
    Args:
    - tokenizer_name (str): Name or path of the tokenizer to use (default: 'gpt2').
    """
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    special_tokens = tokenizer.special_tokens_map
    print("Special Tokens:")
    for token_type, tokens_list in special_tokens.items():
        print(f"{token_type}: {tokens_list}")

def decode_tokenize_text(tokenizer, token_ids, skip_special_tokens=False, remove_pad=True):
    """
    Decodes token IDs into readable text using the provided tokenizer.
     Args:
    - tokenizer (GPT2Tokenizer): The tokenizer object used for encoding.
    - token_ids (Union[int, List[int]]): The token IDs to decode. Can be a single ID or a list of IDs.
    - skip_special_tokens (bool): Whether to skip decoding special tokens (default: True).
    - remove_pad (bool): Whether to remove the [PAD] token from decoded text (default: True).
    Returns:
    - str: The decoded text.
    """
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
    if remove_pad:
        decoded_text = decoded_text.replace(tokenizer.pad_token, "").strip()
    return decoded_text

def tokenize_encoded_texts(tokenizer, encoded_texts, max_length=512):
    """
    Tokenizes a list of encoded texts and ensures each text is padded or truncated to a maximum length.
    """
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_texts = [tokenize_text(tokenizer=tokenizer, text=text, max_length=max_length) for text in encoded_texts]
    return tokenized_texts

def create_dataloader(tokenized_texts, batch_size=4):
    """
    Creates a DataLoader from tokenized texts.
    """
    input_ids = torch.tensor(tokenized_texts, dtype=torch.long)
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

if __name__ == "__main__":
    csv_path = 'cleaned.csv'
    encoded_csv_path = 'csv_encoded.txt'
    batch_size = 4

    # Encode CSV and tokenize it
    encode_csv(csv_path, encoded_csv_path, header=True)
    
    with open(encoded_csv_path, 'r', encoding='utf-8') as f:
        encoded_texts = f.readlines()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set special tokens
    bos_token = "<|startoftext|>"
    eos_token = "<|endoftext|>"


    # Add special tokens to tokenizer
    tokenizer.add_special_tokens({'bos_token': bos_token, 'eos_token': eos_token})

    tokenized_texts = tokenize_encoded_texts(tokenizer=tokenizer, encoded_texts=encoded_texts)
    dataloader = create_dataloader(tokenized_texts, batch_size=batch_size)

    print_special_tokens()
    # Example of iterating through the DataLoader
    for batch in dataloader:
        first_batch_input_ids = batch[0]
        print("Tokenized Line:")
        print(first_batch_input_ids[0])  # Print the first line of tokens in the first batch

        decoded_text = decode_tokenize_text(tokenizer=tokenizer, token_ids=first_batch_input_ids[0])
        print("Decoded Text:")
        print(decoded_text)

        break  # Print only the first batch
        