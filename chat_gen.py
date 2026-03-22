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
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

try:
    import bitsandbytes  # noqa: F401 — check the actual package exists
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


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
        encoded = self.tokenizer(
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
            lambda row: f"{start_token} " + f" {sep_token} ".join(row.astype(str).str.strip()),
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


def _get_free_vram_gb() -> float:
    """Returns free VRAM in GB. Returns 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    reserved = torch.cuda.memory_reserved(0)
    return (total - reserved) / (1024 ** 3)


def _vram_to_batch_size(vram_gb: float) -> int:
    """Pick a batch size based on available VRAM."""
    if vram_gb >= 20:
        return 8
    elif vram_gb >= 14:
        return 4
    elif vram_gb >= 8:
        return 2
    else:
        return 1


def _auto_tune_args(args: Dict[str, Any], dataset_size: int) -> Dict[str, Any]:
    """
    Automatically adjusts max_length, save_every, warmup, and initial batch_size
    based on available VRAM and dataset size.
    """
    args = dict(args)  # don't mutate original
    vram = _get_free_vram_gb()

    auto_batch = _vram_to_batch_size(vram)
    auto_max_len = 256 if vram >= 14 else 128

    # Auto save_every: ~2-3 saves per epoch
    steps_per_epoch = max(1, dataset_size // auto_batch)
    auto_save_every = max(100, steps_per_epoch // 3)

    # Auto warmup: ~3% of first epoch
    auto_warmup = max(50, steps_per_epoch // 30)

    args["batch_size"] = auto_batch
    args["max_length"] = auto_max_len
    args["save_every"] = auto_save_every
    args["warmup_steps"] = auto_warmup

    print(f"\n--- Auto-tuned for {vram:.1f} GB free VRAM ---")
    print(f"  batch_size:  {auto_batch}")
    print(f"  max_length:  {auto_max_len}")
    print(f"  save_every:  {auto_save_every}")
    print(f"  warmup:      {auto_warmup}")
    print()

    return args


def download_base_model(save_directory: str) -> bool:
    """Downloads and saves the base model and tokenizer if not already present."""
    if check_base_model_exists(save_directory):
        print("Base model already exists. Not downloading.")
        return False

    print(f"Downloading {BASE_MODEL_NAME}...")
    device = _get_device()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=_get_dtype(device),
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

    # Load base model — use 4-bit quantization for training to save VRAM (if bitsandbytes installed)
    if for_training and device.type == "cuda" and HAS_BNB:
        print("Loading model with 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    elif for_training and device.type == "cuda" and not HAS_BNB:
        print("Warning: bitsandbytes not installed. Loading in FP16 (uses more VRAM).")
        print("  Install it for lower memory usage: pip install bitsandbytes")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            dtype=dtype,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            dtype=dtype,
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

def create_args(num_epochs: int = 0, batch_size: int = 1, learning_rate: float = 2e-4,
                save_every: int = 500, max_length: int = 512, max_new_tokens: int = 256,
                temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95,
                repetition_penalty: float = 1.2, enableSampleMode: bool = False,
                warmup_steps: int = 100, patience: int = 3, max_epochs: int = 50,
                val_split: float = 0.1) -> Dict[str, Any]:
    """
    num_epochs: 0 = auto-stop (default), any positive number = fixed epochs.
    patience: how many epochs without improvement before auto-stopping.
    max_epochs: upper limit when using auto-stop.
    val_split: fraction of data to use for validation (used by auto-stop).
    """
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
        "warmup_steps": warmup_steps,
        "patience": patience,
        "max_epochs": max_epochs,
        "val_split": val_split
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

def _evaluate(model, dataloader, device, use_amp) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            if use_amp:
                with torch.amp.autocast(device.type):
                    outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
            else:
                outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)

            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def _format_time(seconds: float) -> str:
    """Format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


def train_model(model_directory: str, csv_path: str, args: Optional[Dict[str, Any]] = None) -> None:
    if args is None:
        args = create_args()

    # Determine mode: auto-stop (num_epochs=0) or fixed
    auto_stop = args["num_epochs"] == 0
    max_epochs = args.get("max_epochs", 50) if auto_stop else args["num_epochs"]
    patience = args.get("patience", 3)
    val_split = args.get("val_split", 0.1)

    model, tokenizer = load_model_and_tokenizer(model_directory, for_training=True)
    device = _get_device()
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    # Move to device — quantized models (BNB) are already on CUDA, so .to() is a no-op for them
    try:
        model.to(device)
    except Exception:
        pass  # quantized models may refuse .to(), they're already on the right device

    # Gradient checkpointing: recomputes activations during backward pass instead of storing them
    # Uses less VRAM at the cost of ~20% slower training
    model.gradient_checkpointing_enable()

    # Prepare dataset (lazy tokenization — doesn't load everything into RAM)
    encoded_data = prepare_csv(csv_path, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    if len(encoded_data) == 0:
        print("Error: CSV is empty or has no valid rows. Nothing to train on.")
        return

    # Auto-tune args based on available VRAM and dataset size
    args = _auto_tune_args(args, len(encoded_data))

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.get("learning_rate", 2e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    full_dataset = LazyTokenDataset(
        encoded_data, tokenizer, max_length=args["max_length"],
        sep_token=SPECIAL_TOKENS["sep_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        pad_token=SPECIAL_TOKENS["pad_token"]
    )

    # Split into train/val for early stopping detection
    total_size = len(full_dataset)
    val_size = max(1, int(total_size * val_split)) if auto_stop else 0
    train_size = total_size - val_size

    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices) if val_size > 0 else full_dataset
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=args["batch_size"]) if val_size > 0 else None

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * max_epochs
    warmup_steps = min(args.get("warmup_steps", 100), total_steps // 5)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint if training_state.pt exists
    start_epoch = 0
    training_state_path = os.path.join(model_directory, "training_state.pt")
    if os.path.isfile(training_state_path):
        print("Resuming from training checkpoint...")
        state = torch.load(training_state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if scaler is not None and 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        start_epoch = state.get('epoch', 0)
        if 'rng_state' in state:
            torch.set_rng_state(state['rng_state'])
        if 'cuda_rng_state' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_rng_state'])
        print(f"Resumed from epoch {start_epoch}")

    # Print setup info
    first_sample = full_dataset[0]
    print("First training example (with special tokens):")
    print(decode_data(tokenizer, first_sample['input_ids'], skip_special_tokens=False))
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    if auto_stop:
        print(f"Mode: AUTO-STOP (patience={patience}, max={max_epochs} epochs)")
    else:
        print(f"Mode: FIXED ({max_epochs} epochs)")
    print(f"Steps per epoch: {len(train_loader)}, Warmup: {warmup_steps}")

    # Early stopping state
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    epoch_times = []

    training_start = time.time()
    model.train()

    current_batch_size = args["batch_size"]

    for epoch in range(start_epoch, max_epochs):
        # Re-check VRAM each epoch and adjust batch size if needed
        if device.type == "cuda":
            torch.cuda.empty_cache()
            free_vram = _get_free_vram_gb()
            new_batch = _vram_to_batch_size(free_vram)
            if new_batch != current_batch_size:
                print(f"  [VRAM] {free_vram:.1f} GB free — adjusting batch_size {current_batch_size} -> {new_batch}")
                current_batch_size = new_batch
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
                if val_loader is not None:
                    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=current_batch_size)

        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()

            try:
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
            except torch.cuda.OutOfMemoryError:
                # OOM: clear cache, halve batch size, rebuild dataloader, skip this step
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(f"\n  [OOM] Out of memory! Reducing batch_size to {current_batch_size}")
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
                if val_loader is not None:
                    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=current_batch_size)
                optimizer.zero_grad()
                continue

            scheduler.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / step

            __print_training_progress__(epoch, max_epochs, step, len(train_loader),
                                        loss.item(), avg_loss, training_start, total_steps)

            if step % args["save_every"] == 0:
                _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory, epoch)
                if args.get("enableSampleMode", False):
                    sample_prompts = ["Hello, how are you?", "What's your name?", "Tell me a joke."]
                    sample_prompt = random.choice(sample_prompts)
                    print(f"Prompt: {sample_prompt}")
                    response = generate_responses(model, tokenizer, sample_prompt, device=device, args=args, clean_result=True)
                    print(f"Generated Response: {response}")
                    model.train()

        # Epoch complete
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        num_steps = len(train_loader)
        avg_train_loss = epoch_loss / num_steps if num_steps > 0 else 0.0

        # Validation loss
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device, use_amp)
        else:
            val_loss = avg_train_loss

        # Print epoch summary
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        if val_loader is not None:
            print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Duration:   {_format_time(epoch_duration)}")

        # Time estimate
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        if auto_stop:
            # Estimate: at most (patience - epochs_without_improvement) more epochs if no improvement
            estimated_remaining = patience - epochs_without_improvement
            remaining_time = avg_epoch_time * estimated_remaining
            print(f"  Est. time if no improvement: {_format_time(remaining_time)} ({estimated_remaining} epochs)")
        else:
            remaining_epochs = max_epochs - (epoch + 1)
            remaining_time = avg_epoch_time * remaining_epochs
            print(f"  Est. remaining: {_format_time(remaining_time)} ({remaining_epochs} epochs left)")

        # Early stopping check
        if auto_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                print(f"  >> New best! (val loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  >> No improvement for {epochs_without_improvement}/{patience} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")

        _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory, epoch + 1)

        # Stop if patience exceeded
        if auto_stop and epochs_without_improvement >= patience:
            print(f"\n** Auto-stopped: no improvement for {patience} epochs. Best was epoch {best_epoch} (val loss: {best_val_loss:.4f})")
            print(f"** Your best checkpoint is already saved.")
            break

    total_training_time = time.time() - training_start
    print(f"\nTraining completed in {_format_time(total_training_time)}.")


def _save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, model_directory: str,
                     epoch: int) -> None:
    """Saves LoRA adapter, tokenizer, optimizer/scheduler/RNG state for full resume."""
    adapter_dir = os.path.join(model_directory, "lora_adapter")
    base_dir = os.path.join(model_directory, "base_model")
    training_state_path = os.path.join(model_directory, "training_state.pt")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(base_dir)

    state = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, training_state_path)

    print(f"Checkpoint saved (epoch {epoch})")


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
    encoded = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

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
