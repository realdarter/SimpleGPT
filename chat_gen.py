"""
Coded By Goose
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
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
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


#                      DATASET CLASS

class LazyTokenDataset(Dataset):
    """Memory-efficient dataset that tokenizes on the fly instead of all at once."""
    def __init__(self, texts: List[str], tokenizer, max_length: int,
                 sep_token: str, eos_token: str, pad_token: str) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_id = tokenizer.convert_tokens_to_ids(sep_token)
        self.eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        self.pad_id = tokenizer.convert_tokens_to_ids(pad_token)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Tokenize without padding — we pad manually after appending EOS
        encoded = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_length - 1,  # leave room for EOS
            truncation=True,
            padding=False
        )

        # Append EOS right after the real tokens
        input_ids = encoded['input_ids'] + [self.eos_id]
        attention_mask = [1] * len(input_ids)

        # Now pad on the right to reach max_length
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Mask context (before SEP) and padding with -100
        labels = input_ids.clone()
        sep_positions = (input_ids == self.sep_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            labels[:sep_positions[0].item() + 1] = -100
        labels[input_ids == self.pad_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


#                      UTILITY FUNCTIONS

def ensure_file_exists(file_path: str, create_if_missing: bool = True) -> bool:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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
    """Reads CSV in chunks to avoid loading the entire file into memory at once."""
    start_time = time.time()
    formatted_rows = []
    chunk_size = 10000
    for chunk in pd.read_csv(csv_path, header=0 if header else None, dtype=str, chunksize=chunk_size):
        chunk.fillna('', inplace=True)
        rows = chunk.apply(
            lambda row: f"{start_token} " + f" {sep_token} ".join(row.astype(str).str.strip().str.replace('"', '')),
            axis=1
        )
        formatted_rows.extend(rows.tolist())
    elapsed_time = time.time() - start_time
    print(f"Time taken to prepare CSV: {elapsed_time:.4f} seconds ({len(formatted_rows)} rows)")
    return formatted_rows


def check_base_model_exists(model_path: str) -> bool:
    """Checks if a full base model exists (config.json + weight files)."""
    has_config = os.path.isfile(os.path.join(model_path, 'config.json'))
    has_weights = (
        os.path.isfile(os.path.join(model_path, 'model.safetensors')) or
        os.path.isfile(os.path.join(model_path, 'pytorch_model.bin'))
    )
    return has_config and has_weights


def check_adapter_exists(adapter_path: str) -> bool:
    """Checks if a LoRA adapter exists (adapter_config.json + adapter weights)."""
    has_config = os.path.isfile(os.path.join(adapter_path, 'adapter_config.json'))
    has_weights = (
        os.path.isfile(os.path.join(adapter_path, 'adapter_model.safetensors')) or
        os.path.isfile(os.path.join(adapter_path, 'adapter_model.bin'))
    )
    return has_config and has_weights


#              TOKEN MANAGEMENT FUNCTIONS

def ensure_tokens(model, tokenizer, special_tokens: Dict[str, str] = SPECIAL_TOKENS) -> int:
    """Adds special tokens and resizes embeddings. Returns number of tokens added."""
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def decode_data(tokenizer, token_ids: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


#         MODEL LOADING & DOWNLOADING FUNCTIONS

def _get_device() -> torch.device:
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_dtype(device: torch.device) -> torch.dtype:
    """Returns the appropriate dtype for the device."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def download_base_model(save_directory: str) -> bool:
    """Downloads and saves the base model and tokenizer if not already present."""
    if check_base_model_exists(save_directory):
        print("Base model already exists. Not downloading.")
        return False

    print(f"Downloading {BASE_MODEL_NAME}...")
    device = _get_device()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=_get_dtype(device),
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
    device = _get_device()
    dtype = _get_dtype(device)

    base_model_dir = os.path.join(model_directory, "base_model")
    adapter_dir = os.path.join(model_directory, "lora_adapter")

    # Download base model if needed
    if not check_base_model_exists(base_model_dir) and download:
        download_base_model(base_model_dir)

    if not check_base_model_exists(base_model_dir):
        raise FileNotFoundError(f"Base model not found at {base_model_dir}. Set download=True or provide the model.")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)

    # Load or create LoRA adapter
    if check_adapter_exists(adapter_dir):
        print("Loading existing LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        if for_training:
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
    else:
        print("Warning: No LoRA adapter found. Using base model only.")

    model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model, tokenizer


#         TRAINING ARGUMENTS & PROGRESS FUNCTIONS

def create_args(num_epochs: int = 1, batch_size: int = 1, learning_rate: float = 2e-4,
                save_every: int = 500, max_length: int = 512, max_new_tokens: int = 256,
                temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95,
                repetition_penalty: float = 1.2, enableSampleMode: bool = False,
                warmup_steps: int = 100) -> Dict[str, Any]:
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
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
    device = _get_device()
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.get("learning_rate", 2e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Prepare dataset (lazy tokenization — doesn't load everything into RAM)
    encoded_data = prepare_csv(csv_path, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    dataset = LazyTokenDataset(
        encoded_data, tokenizer, max_length=args["max_length"],
        sep_token=SPECIAL_TOKENS["sep_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        pad_token=SPECIAL_TOKENS["pad_token"]
    )
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * args["num_epochs"]
    warmup_steps = min(args.get("warmup_steps", 100), total_steps // 5)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint if training_state.pt exists
    start_epoch = 0
    start_step = 0
    training_state_path = os.path.join(model_directory, "training_state.pt")
    if os.path.isfile(training_state_path):
        print("Resuming from training checkpoint...")
        state = torch.load(training_state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if scaler is not None and 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        start_epoch = state.get('epoch', 0)
        start_step = state.get('step', 0)
        print(f"Resumed from epoch {start_epoch}, step {start_step}")

    # Print first example for sanity check
    first_sample = dataset[0]
    print("First training example (with special tokens):")
    print(decode_data(tokenizer, first_sample['input_ids'], skip_special_tokens=False))
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    training_start = time.time()
    model.train()

    for epoch in range(start_epoch, args["num_epochs"]):
        epoch_loss = 0.0
        epoch_start = time.time()
        for step, batch in enumerate(dataloader, 1):
            # Skip steps already completed in a resumed epoch
            if epoch == start_epoch and step <= start_step:
                continue
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device.type):
                    outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / step

            __print_training_progress__(epoch, args["num_epochs"], step, len(dataloader),
                                        loss.item(), avg_loss, training_start, total_steps)

            if step % args["save_every"] == 0:
                _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory, epoch, step)
                if args.get("enableSampleMode", False):
                    sample_prompts = ["Hello, how are you?", "What's your name?", "Tell me a joke."]
                    sample_prompt = random.choice(sample_prompts)
                    print(f"Prompt: {sample_prompt}")
                    response = generate_responses(model, tokenizer, sample_prompt, device=device, args=args, clean_result=True)
                    print(f"Generated Response: {response}")
                    model.train()  # switch back after eval in generate_responses

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch+1} duration: {time.time() - epoch_start:.0f} seconds")

        _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory, epoch + 1, 0)

    total_training_time = time.time() - training_start
    print(f"Training completed in {total_training_time // 60:.0f} minutes and {total_training_time % 60:.0f} seconds.")


def _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory: str,
                     epoch: int, step: int) -> None:
    """Saves LoRA adapter, tokenizer, and optimizer/scheduler state for full resume."""
    adapter_dir = os.path.join(model_directory, "lora_adapter")
    base_dir = os.path.join(model_directory, "base_model")
    training_state_path = os.path.join(model_directory, "training_state.pt")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(base_dir)

    state = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, training_state_path)

    print(f"Checkpoint saved (epoch {epoch}, step {step})")


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
                       device: Optional[torch.device] = None,
                       args: Optional[Dict[str, Any]] = None, clean_result: bool = False) -> str:
    if args is None:
        args = create_args()
    if device is None:
        device = _get_device()

    model.eval()

    formatted_prompt = format_prompt(prompt_text, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.get("max_new_tokens", 256),
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
