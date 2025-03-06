"""
Coded By Goose 🪿
Refactored by Assistant

This module provides functions for:
- File and CSV preparation
- Tokenization and special token management
- Model loading and downloading
- Training and generating responses with GPT-2
"""

import os
import time
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#                        GLOBALS & HEADERS

SPECIAL_TOKENS = {
    "pad_token": "<[PAD]>",
    "sep_token": "<[SEP]>",
    "eos_token": "<[EOS]>",
    "bos_token": "<[BOS]>"
}


#                      CUSTOM DATASET CLASS

class CustomDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_masks: torch.Tensor, labels: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


#                      UTILITY FUNCTIONS

def ensure_file_exists(file_path: str, create_if_missing: bool = True) -> bool:
    """
    Ensures that the directory exists and optionally creates an empty file if it does not.
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
    return False


def prepare_csv(csv_path: str, header: bool = True, start_token: str = "", sep_token: str = "") -> List[str]:
    """
    Reads a CSV file and returns a list of formatted strings.
    Each row is prefixed with the start token and columns are joined with the sep token.
    """
    start_time = time.time()
    df = pd.read_csv(csv_path, header=0 if header else None, dtype=str)
    df.fillna('', inplace=True)
    formatted_rows = df.apply(
        lambda row: f"{start_token} " + f" {sep_token} ".join(row.astype(str).str.strip().str.replace('"', '')),
        axis=1
    )
    elapsed_time = time.time() - start_time
    print(f"Time taken to prepare CSV: {elapsed_time:.4f} seconds")
    print(f"First row example: {formatted_rows.iloc[0]}")
    return formatted_rows.tolist()


def check_gpt2_models_exist(model_path: str) -> bool:
    """
    Checks if all necessary GPT-2 model files exist in the specified directory.
    """
    required_files = [
        'config.json',
        'generation_config.json',
        'merges.txt',
        'model.safetensors',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.json'
    ]
    all_exist = True
    for filename in required_files:
        absolute_path = os.path.abspath(os.path.join(model_path, filename))
        if not os.path.isfile(absolute_path):
            print(f"File missing: {absolute_path}")
            all_exist = False
    return all_exist


#                  TOKENIZATION FUNCTIONS

def tokenize_single_text(tokenizer: GPT2Tokenizer, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a single string of text with padding and truncation.
    """
    encoded = tokenizer.encode_plus(
        text.strip(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def tokenize_dataset(tokenizer: GPT2Tokenizer, texts: Union[List[str], str],
                     max_length: int = 512,
                     eos_token: str = SPECIAL_TOKENS["eos_token"]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenizes a list (or single string) of texts, appending the EOS token at the end.
    Returns tensors for input_ids, attention_masks, and labels.
    """
    if isinstance(texts, str):
        texts = [texts]

    tokenized = [tokenizer.encode_plus(text, max_length=max_length, truncation=True, padding='max_length')
                 for text in texts]

    input_ids = torch.tensor([item['input_ids'] for item in tokenized], dtype=torch.long)
    attention_masks = torch.tensor([item['attention_mask'] for item in tokenized], dtype=torch.long)

    # Append EOS token to each sequence
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    eos_tensor = torch.full((input_ids.size(0), 1), eos_id, dtype=torch.long)
    input_ids = torch.cat([input_ids, eos_tensor], dim=1)
    attention_extra = torch.ones((attention_masks.size(0), 1), dtype=torch.long)
    attention_masks = torch.cat([attention_masks, attention_extra], dim=1)

    labels = input_ids.clone()

    # TEMP CODE BELOW
    decoded_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    print("Decoded text from the first tokenized sequence:")
    print(decoded_text)
    # TEMP CODE ABOVE

    return input_ids, attention_masks, labels


#              TOKEN MANAGEMENT FUNCTIONS

def ensure_tokens(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                  special_tokens: Dict[str, str] = SPECIAL_TOKENS) -> None:
    """
    Adds special tokens to the tokenizer and resizes the model's embeddings.
    """
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))


def decode_data(tokenizer: GPT2Tokenizer, token_ids: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True) -> str:
    """
    Decodes token IDs back into a string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


#         MODEL LOADING & DOWNLOADING FUNCTIONS

def download_gpt2_124M(save_directory: str) -> bool:
    """
    Downloads and saves the GPT-2 124M model and tokenizer if not already present.
    """
    if check_gpt2_models_exist(save_directory):
        print("Model already exists. Not downloading.")
        return False

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"GPT-2 model (124M) downloaded and saved in {save_directory}")
    return True


def load_model_and_tokenizer(model_directory: str, download: bool = True) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Loads the GPT-2 model and tokenizer from a directory.
    Downloads the model if necessary.
    """
    start_time = time.time()
    if not check_gpt2_models_exist(model_directory) and download:
        download_gpt2_124M(model_directory)
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model, tokenizer


#         TRAINING ARGUMENTS & PROGRESS FUNCTIONS

def create_args(num_epochs: int = 1, batch_size: int = 1, learning_rate: float = 5e-5,
                save_every: int = 500, max_length: int = 512, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.2) -> Dict[str, Any]:
    """
    Returns a dictionary of training arguments.
    """
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }


def __print_training_progress__(epoch: int, num_epochs: int, step: int, steps_in_epoch: int,
                                loss: float, avg_loss: float, start_time: float, total_steps: int) -> None:
    """
    Prints training progress including loss, elapsed time, and ETA.
    """
    elapsed_time = time.time() - start_time
    steps_completed = epoch * steps_in_epoch + step
    steps_remaining = total_steps - steps_completed
    avg_time_per_step = elapsed_time / steps_completed if steps_completed else 0
    estimated_time_remaining = avg_time_per_step * steps_remaining
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{steps_in_epoch}], "
        f"Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}, Elapsed: {elapsed_time:.2f}s, ETA: {estimated_time_remaining:.2f}s"
    )


#                      TRAINING FUNCTION

def train_model(model_directory: str, csv_path: str, args: Optional[Dict[str, Any]] = None) -> None:
    """
    Trains the GPT-2 model on a dataset provided in a CSV file.
    """
    if args is None:
        args = create_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure special tokens are added
    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.get("learning_rate", 5e-5))
    scaler = torch.amp.GradScaler(device_type='cuda', mean_resizing=False)

    # Prepare dataset
    encoded_data = prepare_csv(csv_path, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, encoded_data, max_length=args["max_length"])
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    print("First training example (with special tokens):")
    print(decode_data(tokenizer, input_ids[0], skip_special_tokens=False))

    total_steps = len(dataloader) * args["num_epochs"]
    training_start = time.time()
    model.train()

    for epoch in range(args["num_epochs"]):
        epoch_loss = 0.0
        epoch_start = time.time()
        for step, batch in enumerate(dataloader, 1):
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / step

            __print_training_progress__(epoch, args["num_epochs"], step, len(dataloader),
                                        loss.item(), avg_loss, training_start, total_steps)

            if step % args["save_every"] == 0:
                model.save_pretrained(model_directory)
                print(f"Model saved at step {step} in epoch {epoch+1}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch+1} duration: {time.time() - epoch_start:.0f} seconds")
        model.save_pretrained(model_directory)

    total_training_time = time.time() - training_start
    print(f"Training completed in {total_training_time // 60:.0f} minutes and {total_training_time % 60:.0f} seconds.")


#                      CLEAN TEXT FUNCTION

def clean_text(uncleaned_text: str, pad_token: str = "", sep_token: str = "",
               eos_token: str = "", bos_token: str = "") -> str:
    """
    Cleans and formats the generated text by removing unwanted tokens.
    """
    special_tokens_dict = {
        'pad_token': pad_token,
        'sep_token': sep_token,
        'eos_token': eos_token,
        'bos_token': bos_token
    }
    # Partition text at the first occurrence of sep_token
    before_sep, sep, after_sep = uncleaned_text.partition(sep_token)
    after_sep = after_sep.replace(bos_token, '').strip()
    while after_sep.startswith(sep_token) or after_sep.startswith(bos_token):
        if after_sep.startswith(sep_token):
            after_sep = after_sep[len(sep_token):].strip()
        if after_sep.startswith(bos_token):
            after_sep = after_sep[len(bos_token):].strip()
    # Take only the first segment after sep_token if multiple exist
    split_text = after_sep.split(sep_token)[0]
    # Remove any remaining special tokens
    for token in special_tokens_dict.values():
        before_sep = before_sep.replace(token, '').strip()
        split_text = split_text.replace(token, '').strip()
    return split_text


#                 PROMPT & GENERATION FUNCTIONS

def format_prompt(prompt_text: str, start_token: str = SPECIAL_TOKENS["bos_token"],
                  sep_token: str = SPECIAL_TOKENS["sep_token"]) -> str:
    """
    Formats a prompt with BOS and SEP tokens.
    """
    return f"{start_token} {prompt_text} {sep_token}"


def generate_responses(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt_text: str,
                       args: Optional[Dict[str, Any]] = None, clean_result: bool = False) -> str:
    """
    Generates a response from the model for a given prompt.
    """
    if args is None:
        args = create_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)

    formatted_prompt = format_prompt(prompt_text, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=args["max_length"],
        temperature=args["temperature"],
        top_k=args["top_k"],
        top_p=args["top_p"],
        repetition_penalty=args["repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    if clean_result:
        generated_text = clean_text(
            generated_text,
            pad_token=tokenizer.pad_token,
            sep_token=tokenizer.sep_token,
            eos_token=tokenizer.eos_token,
            bos_token=tokenizer.bos_token
        )
    return generated_text

