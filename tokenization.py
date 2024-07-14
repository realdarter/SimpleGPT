"""
Coded By Goose 🪿
"""
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader

def tokenize_single_text(tokenizer, text, max_length=512):
    """
    Tokenizes a single line of text and ensures it is padded or truncated to a maximum length.
    Returns a dictionary with 'input_ids' and 'attention_mask'.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        text (str): The text to tokenize.
        max_length (int, optional): The maximum length of the tokenized sequence. Defaults to 512.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask'.
    """
    encoded_text = tokenizer.encode_plus(
        text.strip(), 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length', 
        return_tensors='pt'
    )
    return {
        'input_ids': encoded_text['input_ids'],
        'attention_mask': encoded_text['attention_mask']
    }

def tokenize_dataset(tokenizer, texts, max_length=512, eos_token="<[EOS]>"):
    """
    Tokenizes a list of texts and ensures each text is padded or truncated to a maximum length.
    Returns input_ids, attention_masks, and labels.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        texts (list of str or str): The texts to tokenize.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 512.
        eos_token (str, optional): The end-of-sequence token. Defaults to "<[EOS]>".

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Tensor of token IDs.
            - attention_masks (torch.Tensor): Tensor of attention masks.
            - labels (torch.Tensor): Tensor of labels for language modeling.
    """
    if isinstance(texts, str):
        texts = [texts]
        
    tokenized_texts = [tokenizer.encode_plus(text, max_length=max_length, truncation=True, padding='max_length') for text in texts]
    
    input_ids = torch.tensor([item['input_ids'] for item in tokenized_texts], dtype=torch.long)
    attention_masks = torch.tensor([item['attention_mask'] for item in tokenized_texts], dtype=torch.long)
    
    eos_ids = tokenizer.convert_tokens_to_ids(eos_token)
    eos_tensor = torch.full((input_ids.size(0), 1), eos_ids, dtype=torch.long)
    input_ids = torch.cat([input_ids, eos_tensor], dim=1)
    
    attention_masks = torch.cat([attention_masks, torch.ones((attention_masks.size(0), 1), dtype=torch.long)], dim=1)
    
    labels = input_ids.clone()
    
    return input_ids, attention_masks, labels


def ensure_tokens(tokenizer, pad_token='<[PAD]>', sep_token='<[SEP]>', eos_token='<[EOS]>', bos_token='<[BOS]>'):
    """
    Adds special tokens to the tokenizer if they are not already present.
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to update.
        pad_token (str, optional): The padding token. Defaults to '<[PAD]>'.
        sep_token (str, optional): The separator token. Defaults to '<[SEP]>'.
        eos_token (str, optional): The end-of-sequence token. Defaults to '<[EOS]>'.
        bos_token (str, optional): The beginning-of-sequence token. Defaults to '<[BOS]>'.
    """
    tokenizer.add_special_tokens({'pad_token': pad_token})
    tokenizer.add_special_tokens({'sep_token': sep_token})
    tokenizer.add_special_tokens({'eos_token': eos_token})
    tokenizer.add_special_tokens({'bos_token': bos_token})

def decode_data(tokenizer, token_ids, skip_special_tokens=True):
    """
    Decodes a list or tensor of token IDs back into a string.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        token_ids (list of int or torch.Tensor): The token IDs to decode.
        skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

    Returns:
        str: The decoded string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    
    decoded_data = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return decoded_data

#"""
# Example usage:
model_directory = 'checkpoint/run1'
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
ensure_tokens(tokenizer)

texts = "<[BOS]> Hello 🌍 [⎝oofShorts⎠✔ ᶦˢ ᵃ ⁿᵒᵒᵇ] Hi how are you <[SEP]> I am 😀 good Thanks!"

input_ids, attention_masks, labels = tokenize_dataset(tokenizer, texts)
print(input_ids)
print(attention_masks)
decoded_input = decode_data(tokenizer, input_ids[0].tolist(), skip_special_tokens=False)
print("Decoded Input IDS:", decoded_input)

decoded_labels = decode_data(tokenizer, labels[0].tolist(), skip_special_tokens=False)
print(len(decoded_labels))
print("Decoded Input IDS:", decoded_labels)

#"""