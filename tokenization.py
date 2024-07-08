import torch
from transformers import GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader

def tokenize_single_text(tokenizer, text, max_length=512):
    """
    Tokenizes a single line of text and ensures it is padded or truncated to a maximum length.
    """
    encoded_text = tokenizer.encode_plus(
        text.strip(), 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length', 
        return_tensors='pt'
    )
    return encoded_text

def tokenize_dataset(tokenizer, texts, max_length=512):
    """
    Tokenizes a list of encoded texts and ensures each text is padded or truncated to a maximum length.
    """
    ensure_pad_token(tokenizer)
    tokenized_texts = [tokenize_single_text(tokenizer, text, max_length) for text in texts]
    
    input_ids = torch.cat([item['input_ids'] for item in tokenized_texts], dim=0)
    attention_masks = torch.cat([item['attention_mask'] for item in tokenized_texts], dim=0)
    labels = input_ids.clone()  # Clone input_ids to use as labels
    
    return input_ids, attention_masks, labels

def ensure_pad_token(tokenizer):
    """
    Ensures the tokenizer has a pad token. If not, it adds a pad token.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def create_dataloader(input_ids, attention_masks, batch_size=4):
    """
    Creates a DataLoader from tokenized texts.
    """
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

