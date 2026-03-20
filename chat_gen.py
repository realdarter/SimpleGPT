"""
Coded By Goose 🪿
Refactored by Assistant

This module provides functions for:
- File and CSV preparation
- Tokenization and special token management
- Model loading and downloading
- Training and generating responses with Phi-2 (LoRA fine-tuning)
"""

import os
import time
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
print(torch.__version__)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


BASE_MODEL_NAME = "microsoft/phi-2"

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
    start_time = time.time()
    df = pd.read_csv(csv_path, header=0 if header else None, dtype=str)
    df.fillna('', inplace=True)
    formatted_rows = df.apply(
        lambda row: f"{start_token} " + f" {sep_token} ".join(row.astype(str).str.strip().str.replace('"', '')),
        axis=1
    )
    elapsed_time = time.time() - start_time
    print(f"Time taken to prepare CSV: {elapsed_time:.4f} seconds")
    return formatted_rows.tolist()


def check_model_exists(model_path: str) -> bool:
    """Checks if model files exist in the specified directory."""
    required_files = ['config.json']
    # Check for either safetensors or bin format, or adapter files (LoRA)
    has_weights = (
        os.path.isfile(os.path.join(model_path, 'model.safetensors')) or
        os.path.isfile(os.path.join(model_path, 'pytorch_model.bin')) or
        os.path.isfile(os.path.join(model_path, 'adapter_model.safetensors')) or
        os.path.isfile(os.path.join(model_path, 'adapter_model.bin'))
    )
    has_config = all(
        os.path.isfile(os.path.join(model_path, f)) for f in required_files
    )
    return has_weights and has_config


#                  TOKENIZATION FUNCTIONS

def tokenize_single_text(tokenizer, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
    encoded = tokenizer.encode_plus(
        text.strip(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def tokenize_dataset(tokenizer, texts: Union[List[str], str],
                     max_length: int = 512,
                     eos_token: str = SPECIAL_TOKENS["eos_token"],
                     sep_token: str = SPECIAL_TOKENS["sep_token"]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenizes texts and creates labels that mask the context portion.
    Only the reply (after SEP token) contributes to the loss.
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

    # Create labels: mask context (before SEP) with -100 so only the reply is trained
    labels = input_ids.clone()
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    pad_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad_token"])

    for i in range(labels.size(0)):
        sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            # Mask everything up to and including the SEP token
            mask_end = sep_positions[0].item() + 1
            labels[i, :mask_end] = -100
        # Also mask padding tokens
        labels[i, input_ids[i] == pad_id] = -100

    return input_ids, attention_masks, labels


#              TOKEN MANAGEMENT FUNCTIONS

def ensure_tokens(model, tokenizer, special_tokens: Dict[str, str] = SPECIAL_TOKENS) -> None:
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))


def decode_data(tokenizer, token_ids: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


#         MODEL LOADING & DOWNLOADING FUNCTIONS

def download_base_model(save_directory: str) -> bool:
    """Downloads and saves the base model and tokenizer if not already present."""
    if check_model_exists(save_directory):
        print("Base model already exists. Not downloading.")
        return False

    print(f"Downloading {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"{BASE_MODEL_NAME} downloaded and saved in {save_directory}")
    return True


def load_model_and_tokenizer(model_directory: str, download: bool = True, for_training: bool = False) -> Tuple:
    """
    Loads the model and tokenizer.
    - If a LoRA adapter is found, loads the base model + adapter.
    - If for_training=True, wraps the base model with a new LoRA config.
    """
    start_time = time.time()

    base_model_dir = os.path.join(model_directory, "base_model")
    adapter_dir = os.path.join(model_directory, "lora_adapter")

    # Download base model if needed
    if not check_model_exists(base_model_dir) and download:
        download_base_model(base_model_dir)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)

    # Load or create LoRA adapter
    if check_model_exists(adapter_dir):
        print("Loading existing LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        if for_training:
            # Unfreeze LoRA params for continued training
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
    elif for_training:
        print("Creating new LoRA adapter...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model, tokenizer


#         TRAINING ARGUMENTS & PROGRESS FUNCTIONS

def create_args(num_epochs: int = 1, batch_size: int = 1, learning_rate: float = 2e-4,
                save_every: int = 500, max_length: int = 512, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.2,
                enableSampleMode: bool = False, warmup_steps: int = 100) -> Dict[str, Any]:
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "enableSampleMode": enableSampleMode,
        "warmup_steps": warmup_steps
    }


def __print_training_progress__(epoch: int, num_epochs: int, step: int, steps_in_epoch: int,
                                loss: float, avg_loss: float, start_time: float, total_steps: int) -> None:
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
    if args is None:
        args = create_args()

    model, tokenizer = load_model_and_tokenizer(model_directory, for_training=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.get("learning_rate", 2e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    # Prepare dataset
    encoded_data = prepare_csv(csv_path, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    input_ids, attention_masks, labels = tokenize_dataset(
        tokenizer, encoded_data, max_length=args["max_length"],
        sep_token=tokenizer.sep_token
    )
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * args["num_epochs"]
    warmup_steps = min(args.get("warmup_steps", 100), total_steps // 5)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("First training example (with special tokens):")
    print(decode_data(tokenizer, input_ids[0], skip_special_tokens=False))
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

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

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / step

            __print_training_progress__(epoch, args["num_epochs"], step, len(dataloader),
                                        loss.item(), avg_loss, training_start, total_steps)

            if step % args["save_every"] == 0:
                adapter_dir = os.path.join(model_directory, "lora_adapter")
                model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(os.path.join(model_directory, "base_model"))
                print(f"LoRA adapter saved at step {step} in epoch {epoch+1}")
                if args.get("enableSampleMode", False):
                    sample_prompts = ["Hello, how are you?", "What's your name?", "Tell me a joke."]
                    sample_prompt = random.choice(sample_prompts)
                    print(f"Prompt: {sample_prompt}")
                    response = generate_responses(model, tokenizer, sample_prompt, args=args, clean_result=True)
                    print(f"Generated Response: {response}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch+1} duration: {time.time() - epoch_start:.0f} seconds")

        adapter_dir = os.path.join(model_directory, "lora_adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(os.path.join(model_directory, "base_model"))

    total_training_time = time.time() - training_start
    print(f"Training completed in {total_training_time // 60:.0f} minutes and {total_training_time % 60:.0f} seconds.")


#                      CLEAN TEXT FUNCTION

def clean_text(uncleaned_text: str, pad_token: str = "", sep_token: str = "",
               eos_token: str = "", bos_token: str = "") -> str:
    special_tokens_dict = {
        'pad_token': pad_token,
        'sep_token': sep_token,
        'eos_token': eos_token,
        'bos_token': bos_token
    }
    before_sep, sep, after_sep = uncleaned_text.partition(sep_token)
    after_sep = after_sep.replace(bos_token, '').strip()
    while after_sep.startswith(sep_token) or after_sep.startswith(bos_token):
        if after_sep.startswith(sep_token):
            after_sep = after_sep[len(sep_token):].strip()
        if after_sep.startswith(bos_token):
            after_sep = after_sep[len(bos_token):].strip()
    split_text = after_sep.split(sep_token)[0]
    for token in special_tokens_dict.values():
        before_sep = before_sep.replace(token, '').strip()
        split_text = split_text.replace(token, '').strip()
    return split_text


#                 PROMPT & GENERATION FUNCTIONS

def format_prompt(prompt_text: str, start_token: str = SPECIAL_TOKENS["bos_token"],
                  sep_token: str = SPECIAL_TOKENS["sep_token"]) -> str:
    return f"{start_token} {prompt_text} {sep_token}"


def generate_responses(model, tokenizer, prompt_text: str,
                       args: Optional[Dict[str, Any]] = None, clean_result: bool = False) -> str:
    if args is None:
        args = create_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)
    model.eval()

    formatted_prompt = format_prompt(prompt_text, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args["max_length"],
            temperature=args["temperature"],
            top_k=args["top_k"],
            top_p=args["top_p"],
            repetition_penalty=args["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"]),
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
