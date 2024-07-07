import torch
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader


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

"""
if __name__ == "__main__":
    save_path = 'checkpoint/run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_csv_path = os.path.join(save_path, 'csv_encoded.txt')
    batch_size = 4
    num_epochs = 1
    learning_rate = 5e-5

    ensure_file_exists(csv_path)
    encode_csv(csv_path, encoded_csv_path, header=True)
"""
