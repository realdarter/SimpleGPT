import torch
from transformers import GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader

def tokenize_single_text(tokenizer, text, max_length=512):
    """
    Tokenizes a single line of text and ensures it is padded or truncated to a maximum length.
    Returns a dictionary with 'input_ids' and 'attention_mask'.
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
        'input_ids': encoded_text['input_ids'][0],
        'attention_mask': encoded_text['attention_mask'][0]
    }

def tokenize_dataset(tokenizer, texts, max_length=512):
    """
    Tokenizes a list of encoded texts and ensures each text is padded or truncated to a maximum length.
    """
    if isinstance(texts, str):
        texts = [texts]

    ensure_pad_token(tokenizer)

    tokenized_texts = [tokenize_single_text(tokenizer, text, max_length) for text in texts]
    input_ids = torch.stack([item['input_ids'] for item in tokenized_texts], dim=0)
    attention_masks = torch.stack([item['attention_mask'] for item in tokenized_texts], dim=0)
    labels = torch.tensor([item['input_ids'][1:].tolist() + [tokenizer.eos_token_id] for item in tokenized_texts], dtype=torch.long)

    return input_ids[0], attention_masks[0], labels[0]

def ensure_pad_token(tokenizer, pad_token = "[PAD]", sep_token="<|septext|>", eos_token="<|endoftext|>", bos_token="<|startoftext|>"):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': sep_token})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': eos_token})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': bos_token})

def decode_data(tokenizer, token_ids, skip_special_tokens=True):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    
    decoded_data = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return decoded_data

model_directory = 'checkpoint/run1'
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
texts = "<|startoftext|> Hello Hi how are you <|septext|> I am good Thanks!"

input_ids, attention_masks, labels = tokenize_dataset(tokenizer, texts)

print("Input IDs:", input_ids)
print("Attention Masks:", attention_masks)
print("Labels:", labels)

#print(decode_data(tokenizer, attention_masks))
#print("Attention Masks:", attention_masks)
#print(decode_data(tokenizer, input_ids))
#print(decode_data(tokenizer, labels, skip_special_tokens=False))