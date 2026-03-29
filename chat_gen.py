"""
SimpleGPT — Fine-tuning Phi-3.5 with LoRA for chat generation.
"""

import json
import sys


# --- Colors ---
class _C:
    """ANSI colors for terminal output."""
    if sys.stdout.isatty():
        RESET = '\033[0m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        MAGENTA = '\033[95m'
        GREY = '\033[90m'
    else:
        RESET = BOLD = DIM = GREEN = BLUE = CYAN = YELLOW = RED = MAGENTA = GREY = ''
import os
import platform
import time
import random
import warnings
import inspect
import math

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*not running the flash-attention implementation.*")
warnings.filterwarnings("ignore", message=".*rope_parameters.*original_max_position_embeddings.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*cast from.*")
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.optim import AdamW
from huggingface_hub import snapshot_download
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

try:
    import bitsandbytes  # noqa: F401 — check the actual package exists
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.8)  # Cap VRAM at 80% — leaves room for Windows


BASE_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

SPECIAL_TOKENS = {
    "pad_token": "<[PAD]>",
    "sep_token": "<[SEP]>",
    "eos_token": "<[EOS]>",
    "bos_token": "<[BOS]>"
}

BOT_NAME = "Dara"
_NAME_REPLACEMENTS = {
    "I'm Phi": f"I'm {BOT_NAME}",
    "I am Phi": f"I am {BOT_NAME}",
    "my name is Phi": f"my name is {BOT_NAME}",
    "call me Phi": f"call me {BOT_NAME}",
    "named Phi": f"named {BOT_NAME}",
    "as Phi": f"as {BOT_NAME}",
    "I'm an AI": f"I'm {BOT_NAME}",
    "I am an AI": f"I'm {BOT_NAME}",
    "an AI language model": BOT_NAME,
    "AI language model": BOT_NAME,
    "a language model": "a person",
    "language model": "person",
    "as a chatbot": "",
    "I'm a bot": f"I'm {BOT_NAME}",
    "I am a bot": f"I'm {BOT_NAME}",
    "I'm not a real person": "I'm a real person",
    "I don't have feelings": "I have feelings",
    "I can't feel": "I feel",
    "I don't experience": "I experience",
    "developed by Microsoft": "made by dardar",
    "created by Microsoft": "made by dardar",
    "built by Microsoft": "made by dardar",
    "made by Microsoft": "made by dardar",
    "OpenAI": "dardar",
}


def _apply_bot_name(text: str) -> str:
    """Replace Phi/AI references with the bot name."""
    for old, new in _NAME_REPLACEMENTS.items():
        text = text.replace(old, new)
        text = text.replace(old.lower(), new.lower())
        text = text.replace(old.upper(), new.upper())
    return text


def _recommended_attn_implementation() -> Optional[str]:
    model_name = BASE_MODEL_NAME.lower()
    if "phi-3" in model_name:
        return "eager"
    return None


# --- Dataset ---

class TokenDataset(Dataset):
    """Tokenizes once up front so the training loop does not keep redoing CPU work."""
    def __init__(self, texts: List[str], tokenizer, max_length: int,
                 sep_token: str, eos_token: str, pad_token: str,
                 pretokenize: bool = True, tokenize_batch_size: int = 1024) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_id = tokenizer.convert_tokens_to_ids(sep_token)
        self.eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        self.pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        self.samples: Optional[List[Dict[str, torch.Tensor]]] = None
        self.lengths: List[int] = []

        if pretokenize:
            self.samples = self._pretokenize(tokenize_batch_size)

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.texts)

    def _encode_token_ids(self, token_ids: List[int]) -> Optional[Dict[str, torch.Tensor]]:
        token_ids = token_ids[: self.max_length - 1]
        if not token_ids:
            return None

        try:
            sep_index = token_ids.index(self.sep_id)
        except ValueError:
            return None

        # If truncation removed the reply, skip the row instead of training on prompt-only text.
        if sep_index >= len(token_ids) - 1:
            return None

        input_ids = token_ids + [self.eos_id]
        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        labels[:sep_index + 1] = [-100] * (sep_index + 1)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.max_length - 1,
            truncation=True,
            padding=False
        )
        sample = self._encode_token_ids(encoded['input_ids'])
        if sample is None:
            raise ValueError("Training sample lost its separator or reply after truncation.")
        return sample

    def _pretokenize(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        start_time = time.time()
        samples: List[Dict[str, torch.Tensor]] = []
        lengths: List[int] = []
        dropped_samples = 0

        for start_idx in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[start_idx:start_idx + batch_size]
            encoded_batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length - 1,
                truncation=True,
                padding=False
            )
            for token_ids in encoded_batch['input_ids']:
                sample = self._encode_token_ids(token_ids)
                if sample is None:
                    dropped_samples += 1
                    continue
                samples.append(sample)
                lengths.append(int(sample['input_ids'].size(0)))

        elapsed = time.time() - start_time
        self.lengths = lengths
        print(f"{_C.DIM}Pretokenized {len(samples)} samples in {elapsed:.2f} seconds.{_C.RESET}")
        if dropped_samples:
            print(f"Skipped {dropped_samples} truncated sample(s) that no longer contained a reply.")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.samples is not None:
            return self.samples[idx]
        return self._encode_text(self.texts[idx])


class LengthBucketBatchSampler(BatchSampler):
    """Groups similar-length samples together to reduce padding waste."""
    def __init__(self, indices: List[int], lengths: List[int], batch_size: int,
                 shuffle: bool = True, bucket_size_multiplier: int = 50) -> None:
        self.indices = list(indices)
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size_multiplier = max(2, bucket_size_multiplier)

    def __iter__(self):
        indices = list(self.indices)
        if self.shuffle:
            random.shuffle(indices)

        bucket_size = max(self.batch_size, self.batch_size * self.bucket_size_multiplier)
        batches: List[List[int]] = []

        for start_idx in range(0, len(indices), bucket_size):
            bucket = indices[start_idx:start_idx + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            bucket_batches = [
                bucket[i:i + self.batch_size]
                for i in range(0, len(bucket), self.batch_size)
                if bucket[i:i + self.batch_size]
            ]
            if self.shuffle:
                random.shuffle(bucket_batches)
            batches.extend(bucket_batches)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)


class DynamicPaddingCollator:
    """Pads each batch to its own max length instead of always using the global max."""
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(sample['input_ids'].size(0) for sample in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for row_idx, sample in enumerate(batch):
            seq_len = sample['input_ids'].size(0)
            input_ids[row_idx, :seq_len] = sample['input_ids']
            attention_mask[row_idx, :seq_len] = sample['attention_mask']
            labels[row_idx, :seq_len] = sample['labels']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# --- Utilities ---

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


def _normalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna('')
        .astype(str)
        .str.replace('\r\n', '\n', regex=False)
        .str.replace('\r', '\n', regex=False)
        .str.strip()
    )


def prepare_csv(csv_path: str, header: bool = True, start_token: str = "", sep_token: str = "") -> List[str]:
    """Reads CSV in chunks and formats chat training rows efficiently."""
    start_time = time.time()
    formatted_rows = []
    chunk_size = 10000
    for chunk in pd.read_csv(csv_path, header=0 if header else None, dtype=str, chunksize=chunk_size):
        if {'context', 'reply'}.issubset(chunk.columns):
            context = _normalize_series(chunk['context'])
            reply = _normalize_series(chunk['reply'])
            valid_mask = (context != '') & (reply != '')
            rows = (
                start_token + " " +
                context[valid_mask] +
                f" {sep_token} " +
                reply[valid_mask]
            )
            formatted_rows.extend(rows.tolist())
            continue

        chunk.fillna('', inplace=True)
        rows = chunk.apply(
            lambda row: f"{start_token} " + f" {sep_token} ".join(row.astype(str).str.strip()),
            axis=1
        )
        formatted_rows.extend(row for row in rows.tolist() if row.strip() != start_token.strip())
    elapsed_time = time.time() - start_time
    print(f"{_C.DIM}Time taken to prepare CSV: {elapsed_time:.4f} seconds ({len(formatted_rows)} rows){_C.RESET}")
    return formatted_rows


def _filter_formatted_rows_for_training(formatted_rows: List[str], tokenizer, max_length: int,
                                        sep_token: str, tokenize_batch_size: int = 1024) -> Tuple[List[str], Dict[str, int]]:
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    if sep_id is None or sep_id < 0:
        raise ValueError("Tokenizer separator token is not configured.")

    kept_rows: List[str] = []
    dropped_missing_sep = 0
    dropped_missing_target = 0

    for start_idx in range(0, len(formatted_rows), tokenize_batch_size):
        batch_rows = formatted_rows[start_idx:start_idx + tokenize_batch_size]
        encoded_batch = tokenizer(
            batch_rows,
            max_length=max_length - 1,
            truncation=True,
            padding=False
        )
        for row_text, token_ids in zip(batch_rows, encoded_batch["input_ids"]):
            try:
                sep_index = token_ids.index(sep_id)
            except ValueError:
                dropped_missing_sep += 1
                continue

            if sep_index >= len(token_ids) - 1:
                dropped_missing_target += 1
                continue

            kept_rows.append(row_text)

    return kept_rows, {
        "rows_kept": len(kept_rows),
        "rows_dropped_missing_sep": dropped_missing_sep,
        "rows_dropped_missing_target": dropped_missing_target,
        "rows_dropped_total": dropped_missing_sep + dropped_missing_target,
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=True, indent=2)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=True) + "\n")


def _append_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _make_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def _prepare_training_log_paths(model_directory: str, run_id: str,
                                existing_log_dir: Optional[str] = None) -> Dict[str, str]:
    logs_root = os.path.join(model_directory, "training_logs")
    run_dir = existing_log_dir or os.path.join(logs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    paths = {
        "logs_root": logs_root,
        "run_dir": run_dir,
        "metadata": os.path.join(run_dir, "run_metadata.json"),
        "summary": os.path.join(run_dir, "summary.json"),
        "events": os.path.join(run_dir, "events.jsonl"),
        "steps": os.path.join(run_dir, "steps.jsonl"),
        "epochs": os.path.join(run_dir, "epochs.jsonl"),
        "samples": os.path.join(run_dir, "samples.jsonl"),
        "samples_text": os.path.join(run_dir, "samples.txt"),
    }
    _write_json(
        os.path.join(logs_root, "latest_run.json"),
        {
            "updated_at": _now_iso(),
            "run_id": run_id,
            "run_dir": run_dir,
        },
    )
    return paths


def _truncate_text(text: str, limit: int = 300) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _extract_prompt_and_target(formatted_text: str, bos_token: str,
                               sep_token: str, eos_token: str) -> Optional[Dict[str, str]]:
    if not formatted_text:
        return None

    text = formatted_text.replace(bos_token, "").strip()
    prompt, sep, target = text.partition(sep_token)
    if not sep:
        return None

    prompt = prompt.strip()
    target = target.replace(eos_token, "").strip()
    if not prompt or not target:
        return None

    return {
        "prompt": prompt,
        "target": target,
    }


def _select_sample_pairs(formatted_rows: List[str], indices: List[int], tokenizer,
                         sample_count: int, split_name: str) -> List[Dict[str, Any]]:
    sample_pairs: List[Dict[str, Any]] = []
    if sample_count <= 0:
        return sample_pairs

    for source_index in indices:
        pair = _extract_prompt_and_target(
            formatted_rows[source_index],
            tokenizer.bos_token,
            tokenizer.sep_token,
            tokenizer.eos_token,
        )
        if pair is None:
            continue

        sample_pairs.append(
            {
                "split": split_name,
                "source_index": int(source_index),
                "prompt": pair["prompt"],
                "target": pair["target"],
            }
        )
        if len(sample_pairs) >= sample_count:
            break

    return sample_pairs


def _get_cuda_runtime_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    free_bytes, _ = torch.cuda.mem_get_info()
    return {
        "cuda_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 4),
        "cuda_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 4),
        "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 4),
        "cuda_free_gb": round(free_bytes / (1024 ** 3), 4),
    }


def _get_environment_diagnostics(device: torch.device) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "has_bitsandbytes": HAS_BNB,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        diagnostics["gpu"] = {
            "name": props.name,
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "multi_processor_count": getattr(props, "multi_processor_count", None),
            "major": getattr(props, "major", None),
            "minor": getattr(props, "minor", None),
            "usable_free_vram_gb": round(_get_free_vram_gb(), 2),
        }
        diagnostics.update(_get_cuda_runtime_stats())

    return diagnostics


def check_base_model_exists(model_path: str) -> bool:
    """Checks if a full base model exists (config.json + weight files)."""
    has_config = os.path.isfile(os.path.join(model_path, 'config.json'))
    weight_filenames = os.listdir(model_path) if os.path.isdir(model_path) else []
    has_weights = any(
        filename == 'model.safetensors' or
        filename == 'pytorch_model.bin' or
        filename == 'model.safetensors.index.json' or
        filename == 'pytorch_model.bin.index.json' or
        filename.endswith('.safetensors') or
        filename.endswith('.bin')
        for filename in weight_filenames
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


def _archive_existing_path(path: str, suffix: str) -> Optional[str]:
    """Renames an existing file/dir so incompatible state is preserved instead of deleted."""
    if not os.path.exists(path):
        return None

    parent = os.path.dirname(path)
    name = os.path.basename(path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = os.path.join(parent, f"{name}.{suffix}.{timestamp}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{name}.{suffix}.{timestamp}-{counter}")
        counter += 1

    os.replace(path, candidate)
    return candidate


# --- Token Management ---

def ensure_tokens(model, tokenizer, special_tokens: Dict[str, str] = SPECIAL_TOKENS) -> int:
    """Adds special tokens and resizes embeddings. Returns number of tokens added."""
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        current = model.get_input_embeddings().weight.shape[0]
        target = max(current, len(tokenizer))
        model.resize_token_embeddings(target)
    return num_added


def decode_data(tokenizer, token_ids: Union[List[int], torch.Tensor],
                skip_special_tokens: bool = True) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


# --- Model Loading ---

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


VRAM_RESERVE_FRACTION = 0.20  # keep 20% VRAM free — prevents system lag on single-GPU Windows


def _get_free_vram_gb() -> float:
    """Returns usable free VRAM in GB (reserves 10% for system). Returns 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        system_reserve = total_bytes * VRAM_RESERVE_FRACTION
        usable = free_bytes - system_reserve
    except RuntimeError:
        props = torch.cuda.get_device_properties(0)
        total_bytes = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        reserved_by_torch = torch.cuda.memory_reserved(0)
        system_reserve = total_bytes * VRAM_RESERVE_FRACTION
        usable = total_bytes - reserved_by_torch - system_reserve
    return max(0.0, usable / (1024 ** 3))


def _round_sequence_cap(length: int) -> int:
    caps = (64, 96, 128, 160, 192, 224, 256, 320, 384, 512)
    for cap in caps:
        if length <= cap:
            return cap
    return 512


def _estimate_length_stats(texts: List[str], tokenizer, sample_size: int = 4096) -> Dict[str, float]:
    if not texts:
        return {"mean": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "p995": 0.0, "max": 0.0}

    sample_count = min(len(texts), sample_size)
    sample = random.sample(texts, sample_count) if len(texts) > sample_count else texts
    encoded = tokenizer(sample, truncation=False, padding=False)
    lengths = sorted(len(token_ids) + 1 for token_ids in encoded['input_ids'])  # +1 for manual EOS

    def quantile(q: float) -> float:
        idx = min(len(lengths) - 1, max(0, math.ceil(q * len(lengths)) - 1))
        return float(lengths[idx])

    return {
        "mean": float(sum(lengths) / len(lengths)),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "p995": quantile(0.995),
        "max": float(lengths[-1]),
    }


def _vram_to_batch_size(vram_gb: float, max_length: int) -> int:
    """Pick a micro-batch size from VRAM and the chosen sequence cap."""
    if vram_gb >= 10:
        batch_size = 16
    elif vram_gb >= 8:
        batch_size = 12
    elif vram_gb >= 6:
        batch_size = 8
    elif vram_gb >= 4:
        batch_size = 4
    elif vram_gb >= 3:
        batch_size = 2
    else:
        batch_size = 1

    if max_length > 384:
        batch_size = max(1, batch_size // 4)
    elif max_length > 256:
        batch_size = max(1, batch_size // 2)
    elif max_length > 160:
        batch_size = max(1, (batch_size * 3) // 4)

    return max(1, int(batch_size))


def _auto_tune_args(args: Dict[str, Any], dataset_size: int,
                    length_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Automatically adjusts sequence cap, micro-batch size, gradient accumulation,
    and warmup from the dataset/token stats and available VRAM.
    """
    args = dict(args)  # don't mutate original
    vram = _get_free_vram_gb()
    user_max_length = int(args.get("max_length", 512))
    target_effective_batch_size = max(1, int(args.get("target_effective_batch_size", 32)))
    length_stats = length_stats or {}

    suggested_cap = int(length_stats.get("p995", 0) * 1.15) if length_stats else 0
    if suggested_cap <= 0:
        suggested_cap = 128 if vram >= 6 else 64
    auto_max_len = min(user_max_length, _round_sequence_cap(max(64, suggested_cap)))

    auto_batch = _vram_to_batch_size(vram, auto_max_len)
    auto_grad_accum = max(1, math.ceil(target_effective_batch_size / auto_batch))
    effective_batch = auto_batch * auto_grad_accum
    steps_per_epoch = max(1, math.ceil(dataset_size / auto_batch))
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / auto_grad_accum))
    auto_warmup = max(25, optimizer_steps_per_epoch // 20)

    args["batch_size"] = auto_batch
    args["max_length"] = auto_max_len
    if args.get("gradient_accumulation_steps") is None:
        args["gradient_accumulation_steps"] = auto_grad_accum
    args["warmup_steps"] = auto_warmup

    print(f"{_C.BLUE}{_C.BOLD}\n--- Auto-tuned for {vram:.1f} GB free VRAM ---{_C.RESET}")
    if length_stats:
        print(
            "  token stats: "
            f"mean={length_stats['mean']:.1f}, "
            f"p95={length_stats['p95']:.0f}, "
            f"p99={length_stats['p99']:.0f}, "
            f"p99.5={length_stats['p995']:.0f}"
        )
    print(f"{_C.DIM}  micro_batch: {auto_batch}{_C.RESET}")
    print(f"{_C.DIM}  max_length:  {auto_max_len}{_C.RESET}")
    print(f"{_C.DIM}  grad_accum:  {args['gradient_accumulation_steps']}{_C.RESET}")
    print(f"{_C.DIM}  eff_batch:   {effective_batch}{_C.RESET}")
    print(f"{_C.DIM}  save_every:  {args['save_every']}{_C.RESET}")
    print(f"{_C.DIM}  warmup:      {auto_warmup}{_C.RESET}")
    print()

    return args


def download_base_model(save_directory: str) -> bool:
    """Downloads and saves the base model and tokenizer if not already present."""
    if check_base_model_exists(save_directory):
        print("Base model already exists. Not downloading.")
        return False

    print(f"{_C.CYAN}Downloading {BASE_MODEL_NAME}...{_C.RESET}")
    os.makedirs(save_directory, exist_ok=True)

    # Download the original repository files directly instead of loading the model and
    # re-saving it. Phi-3.5 can trip a tied-weights bug in Transformers during save_pretrained.
    snapshot_download(
        repo_id=BASE_MODEL_NAME,
        local_dir=save_directory,
        token=os.getenv("HF_TOKEN"),
        allow_patterns=[
            "*.json",
            "*.model",
            "*.txt",
            "*.safetensors",
            "*.bin",
        ],
    )

    if not check_base_model_exists(save_directory):
        raise FileNotFoundError(
            f"Downloaded files for {BASE_MODEL_NAME}, but no model weights were found in {save_directory}."
        )

    # Remove auto_map so we use native transformers Phi-3 support (not outdated custom code)
    config_path = os.path.join(save_directory, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if "auto_map" in cfg:
                del cfg["auto_map"]
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        except Exception:
            pass

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
    attn_implementation = _recommended_attn_implementation()
    load_notes: List[str] = []

    base_model_dir = os.path.join(model_directory, "base_model")
    adapter_dir = os.path.join(model_directory, "lora_adapter")
    best_adapter_dir = os.path.join(model_directory, "best_lora_adapter")

    # Download base model if needed
    if not check_base_model_exists(base_model_dir) and download:
        download_base_model(base_model_dir)

    if not check_base_model_exists(base_model_dir):
        raise FileNotFoundError(f"Base model not found at {base_model_dir}. Set download=True or provide the model.")

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = attn_implementation

    # Load base model — use quantization when possible to reduce VRAM / host-memory pressure.
    # low_cpu_mem_usage=True loads weights shard by shard instead of all at once.
    if for_training and device.type == "cuda" and HAS_BNB:
        load_notes.append("mode=4bit-train")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            quantization_config=bnb_config,
            **model_load_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
    elif not for_training and device.type == "cuda" and HAS_BNB:
        load_notes.append("mode=4bit-infer")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            quantization_config=bnb_config,
            **model_load_kwargs,
        )
    elif for_training and device.type == "cuda" and not HAS_BNB:
        load_notes.append("mode=fp16-train")
        print(f"{_C.YELLOW}Warning: bitsandbytes not installed. Loading in FP16 (uses more VRAM).{_C.RESET}")
        print("  Install it for lower memory usage: pip install bitsandbytes")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            dtype=dtype,
            **model_load_kwargs,
        )
    else:
        load_notes.append(f"mode={str(dtype).replace('torch.', '')}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            dtype=dtype,
            **model_load_kwargs,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=False)

    # Add special tokens and resize embeddings BEFORE loading adapter
    # so the embedding size matches what the adapter was trained with
    load_notes.append(f"base={base_model_dir}")

    # Map our custom tokens to Phi-3.5's native tokens for smart initialization
    # Instead of random/mean init, we transfer knowledge from tokens that serve the same purpose
    _INIT_FROM = {
        "<[PAD]>": ["<|endoftext|>"],                  # padding
        "<[SEP]>": ["<|end|>", "<|assistant|>"],       # user->assistant transition
        "<[EOS]>": ["<|end|>"],                        # end of sequence
        "<[BOS]>": ["<|user|>"],                       # start of user input
    }

    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    target_vocab_size = max(current_vocab_size, len(tokenizer))  # never shrink embeddings

    if current_vocab_size != target_vocab_size:
        load_notes.append(f"resize={current_vocab_size}->{target_vocab_size}")
        old_embeddings = model.get_input_embeddings().weight.data.clone()
        model.resize_token_embeddings(target_vocab_size)

        if num_added > 0:
            with torch.no_grad():
                new_embeddings = model.get_input_embeddings().weight.data
                for token_name, token_value in SPECIAL_TOKENS.items():
                    new_id = tokenizer.convert_tokens_to_ids(token_value)
                    source_names = _INIT_FROM.get(token_value, [])
                    source_ids = [tokenizer.convert_tokens_to_ids(s) for s in source_names
                                  if tokenizer.convert_tokens_to_ids(s) < old_embeddings.shape[0]]

                    if source_ids:
                        # Average the embeddings of semantically related native tokens
                        source_embeds = torch.stack([old_embeddings[sid] for sid in source_ids])
                        new_embeddings[new_id] = source_embeds.mean(dim=0)
                        pass
                    else:
                        # Fallback to mean of all embeddings
                        new_embeddings[new_id] = old_embeddings.mean(dim=0)

    # Load or create LoRA adapter
    inference_adapter_dir = adapter_dir
    if not for_training and check_adapter_exists(best_adapter_dir):
        inference_adapter_dir = best_adapter_dir

    if check_adapter_exists(inference_adapter_dir):
        adapter_label = os.path.basename(inference_adapter_dir)
        load_notes.append(f"adapter={adapter_label}")
        try:
            model = PeftModel.from_pretrained(model, inference_adapter_dir)
            if for_training:
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
        except RuntimeError as e:
            if "size mismatch" in str(e):
                training_state = os.path.join(model_directory, "training_state.pt")
                if for_training:
                    print(f"{_C.YELLOW}Warning: Existing LoRA adapter is incompatible with the current tokenizer/model shape.{_C.RESET}")
                    archived_adapter = _archive_existing_path(adapter_dir, "incompatible")
                    archived_state = _archive_existing_path(training_state, "incompatible")
                    if archived_adapter:
                        print(f"Archived incompatible adapter to: {archived_adapter}")
                    if archived_state:
                        print(f"Archived incompatible training state to: {archived_state}")
                    print("Creating a fresh LoRA adapter for training.")
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05,
                        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
                        bias="none"
                    )
                    model = get_peft_model(model, lora_config)
                    model.print_trainable_parameters()
                else:
                    print(f"{_C.YELLOW}Warning: Existing LoRA adapter is incompatible with the current tokenizer/model shape.{_C.RESET}")
                    print("Keeping the incompatible adapter on disk and using the base model for inference.")
            else:
                raise
    elif for_training:
        load_notes.append("adapter=new")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        load_notes.append("adapter=base-only")
        print(f"{_C.YELLOW}Warning: No LoRA adapter found. Using base model only.{_C.RESET}")

    summary = " | ".join(load_notes)
    print(f"{_C.GREEN}{_C.BOLD}Model load summary: {summary} | loaded_in={time.time() - start_time:.2f}s{_C.RESET}")
    return model, tokenizer


# --- Training Arguments ---

def create_args(num_epochs: int = 0, batch_size: int = 1, learning_rate: float = 2e-4,
                save_every: int = 500, max_length: int = 512, max_new_tokens: int = 256,
                temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95,
                repetition_penalty: float = 1.2, enableSampleMode: bool = False,
                warmup_steps: int = 100, patience: int = 3, max_epochs: int = 50,
                val_split: float = 0.1, log_every: int = 50,
                pretokenize: bool = True, num_workers: Optional[int] = None,
                gradient_checkpointing: Optional[bool] = None,
                gradient_accumulation_steps: Optional[int] = None,
                target_effective_batch_size: int = 32,
                max_eval_samples: int = 2048,
                length_bucketing: bool = True,
                min_improvement: float = 0.0,
                seed: int = 42,
                write_diagnostics: bool = True,
                data_preview_count: int = 3,
                sample_preview_count: int = 3,
                sample_log_every_epochs: int = 1,
                sample_max_new_tokens: int = 96,
                max_newlines: int = 2,
                gpu_throttle: float = 0.0) -> Dict[str, Any]:
    """
    num_epochs: 0 = auto-stop (default), any positive number = fixed epochs.
    patience: how many epochs without improvement before auto-stopping.
    max_epochs: upper limit when using auto-stop.
    val_split: fraction of data to use for validation (used by auto-stop).
    gradient_accumulation_steps: accumulate gradients over N micro-batches before update.
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
        "val_split": val_split,
        "log_every": log_every,
        "pretokenize": pretokenize,
        "num_workers": num_workers,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "target_effective_batch_size": target_effective_batch_size,
        "max_eval_samples": max_eval_samples,
        "length_bucketing": length_bucketing,
        "min_improvement": min_improvement,
        "seed": seed,
        "write_diagnostics": write_diagnostics,
        "data_preview_count": data_preview_count,
        "sample_preview_count": sample_preview_count,
        "sample_log_every_epochs": sample_log_every_epochs,
        "sample_max_new_tokens": sample_max_new_tokens,
        "max_newlines": max_newlines,
        "gpu_throttle": gpu_throttle,
    }


def __print_training_progress__(epoch: int, num_epochs: int, step: int, steps_in_epoch: int,
                                loss: float, avg_loss: float, start_time: float, total_steps: int,
                                lr: float = 0.0) -> None:
    elapsed_time = time.time() - start_time
    steps_completed = epoch * steps_in_epoch + step
    steps_remaining = total_steps - steps_completed
    avg_time_per_step = elapsed_time / steps_completed if steps_completed else 0
    estimated_time_remaining = avg_time_per_step * steps_remaining
    loss_color = _C.GREEN if loss < avg_loss else _C.YELLOW if loss < avg_loss * 1.2 else _C.RED
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Step [{step}/{steps_in_epoch}] "
        f"{loss_color}Loss: {loss:.4f}{_C.RESET} "
        f"{_C.CYAN}Avg: {avg_loss:.4f}{_C.RESET} "
        f"LR: {lr:.2e} "
        f"Elapsed: {_format_time(elapsed_time)} "
        f"ETA: {_format_time(estimated_time_remaining)}"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_num_workers(args: Dict[str, Any], device: torch.device) -> int:
    configured = args.get("num_workers")
    if configured is not None:
        return max(0, int(configured))
    if args.get("pretokenize", True):
        return 0
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 2
    return min(4, max(1, cpu_count // 4))


def _build_dataloader(dataset, batch_size: int, shuffle: bool, collate_fn,
                      num_workers: int, pin_memory: bool,
                      lengths: Optional[List[int]] = None,
                      indices: Optional[List[int]] = None,
                      use_length_bucketing: bool = False) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    if use_length_bucketing and lengths is not None and indices is not None:
        loader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            indices=list(indices),
            lengths=lengths,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(**loader_kwargs)


def _write_sample_records(log_paths: Dict[str, str], records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    for record in records:
        _append_jsonl(log_paths["samples"], record)

        text_block = [
            f"[{record['timestamp']}] epoch={record['epoch']} split={record['split']} source_index={record['source_index']}",
            f"prompt: {record['prompt']}",
            f"target: {record['target']}",
            f"generated: {record['generated']}",
            "",
        ]
        _append_text(log_paths["samples_text"], "\n".join(text_block))


def _generate_diagnostic_samples(model, tokenizer, sample_pairs: List[Dict[str, Any]],
                                 device: torch.device, args: Dict[str, Any],
                                 epoch: int, log_paths: Dict[str, str]) -> None:
    if not sample_pairs:
        return

    sample_args = dict(args)
    sample_args["max_new_tokens"] = min(
        int(args.get("max_new_tokens", 256)),
        max(16, int(args.get("sample_max_new_tokens", 96))),
    )

    original_use_cache = None
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        original_use_cache = model.config.use_cache
        model.config.use_cache = True

    records: List[Dict[str, Any]] = []
    try:
        for pair in sample_pairs:
            generated = generate_responses(
                model,
                tokenizer,
                pair["prompt"],
                device=device,
                args=sample_args,
                clean_result=True,
            )
            records.append(
                {
                    "timestamp": _now_iso(),
                    "epoch": epoch,
                    "split": pair["split"],
                    "source_index": pair["source_index"],
                    "prompt": _truncate_text(pair["prompt"], 500),
                    "target": _truncate_text(pair["target"], 500),
                    "generated": _truncate_text(generated, 800),
                    "sample_max_new_tokens": sample_args["max_new_tokens"],
                }
            )
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(was_training)

    _write_sample_records(log_paths, records)


# --- Training ---

def _evaluate(model, dataloader, device, use_amp, non_blocking: bool = False) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch = batch['input_ids'].to(device, non_blocking=non_blocking)
            attention_mask_batch = batch['attention_mask'].to(device, non_blocking=non_blocking)
            labels_batch = batch['labels'].to(device, non_blocking=non_blocking)

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


def train_model(model_directory: str, csv_path: str, args: Optional[Dict[str, Any]] = None,
                on_event: Optional[callable] = None) -> None:
    if args is None:
        args = create_args()

    def _emit(event_type: str, **kwargs):
        if on_event is not None:
            on_event(event_type, **kwargs)

    user_args = dict(args)
    _seed_everything(int(args.get("seed", 42)))

    # Determine mode: auto-stop (num_epochs=0) or fixed
    auto_stop = args["num_epochs"] == 0
    max_epochs = args.get("max_epochs", 50) if auto_stop else args["num_epochs"]
    patience = args.get("patience", 3)
    val_split = args.get("val_split", 0.1)
    min_improvement = float(args.get("min_improvement", 0.0))
    max_eval_samples = max(0, int(args.get("max_eval_samples", 2048)))

    model, tokenizer = load_model_and_tokenizer(model_directory, for_training=True)
    device = _get_device()
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"
    print(f"{_C.CYAN}Using device: {device}{_C.RESET}")

    # Special tokens + device placement already handled by load_model_and_tokenizer (device_map="auto")

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Prepare dataset text first so we can tune around the real training set size.
    encoded_data = prepare_csv(csv_path, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    if len(encoded_data) == 0:
        print(f"{_C.RED}Error: CSV is empty or has no valid rows. Nothing to train on.{_C.RESET}")
        return

    # Flush cache so VRAM measurement is accurate after model load
    if device.type == "cuda":
        torch.cuda.empty_cache()

    length_stats = _estimate_length_stats(encoded_data, tokenizer)
    args = _auto_tune_args(args, len(encoded_data), length_stats)
    encoded_data, truncation_filter_stats = _filter_formatted_rows_for_training(
        encoded_data,
        tokenizer,
        int(args["max_length"]),
        tokenizer.sep_token,
    )
    if truncation_filter_stats["rows_dropped_total"]:
        print(
            "Dropped "
            f"{truncation_filter_stats['rows_dropped_total']} sample(s) after truncation "
            "because the separator or reply would have been removed."
        )
    if len(encoded_data) == 0:
        print(f"{_C.RED}Error: all rows were truncated into invalid prompt-only samples. Increase max_length.{_C.RESET}")
        return
    log_every = max(1, int(args.get("log_every", 50)))
    write_diagnostics = bool(args.get("write_diagnostics", True))
    data_preview_count = max(0, int(args.get("data_preview_count", 3)))
    sample_preview_count = max(0, int(args.get("sample_preview_count", 3)))
    sample_log_every_epochs = max(0, int(args.get("sample_log_every_epochs", 1)))
    num_workers = _resolve_num_workers(args, device)
    grad_accum_steps = max(1, int(args.get("gradient_accumulation_steps", 1)))
    current_batch_size = int(args["batch_size"])
    target_effective_batch_size = max(1, int(args.get("target_effective_batch_size", current_batch_size * grad_accum_steps)))

    use_gradient_checkpointing = args.get("gradient_checkpointing")
    if use_gradient_checkpointing is None:
        use_gradient_checkpointing = _get_free_vram_gb() < 6
    use_gradient_checkpointing = bool(use_gradient_checkpointing)
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    print(f"{_C.DIM}Gradient checkpointing: {'enabled' if use_gradient_checkpointing else 'disabled'}{_C.RESET}")

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_kwargs = {
        "lr": args.get("learning_rate", 2e-4),
        "weight_decay": 0.01
    }
    optimizer = None
    if device.type == "cuda" and "fused" in inspect.signature(AdamW).parameters:
        try:
            optimizer = AdamW(trainable_params, fused=True, **optimizer_kwargs)
            print(f"{_C.DIM}Using fused AdamW.{_C.RESET}")
        except (RuntimeError, TypeError):
            optimizer = None
    if optimizer is None:
        optimizer = AdamW(trainable_params, **optimizer_kwargs)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    full_dataset = TokenDataset(
        encoded_data, tokenizer, max_length=args["max_length"],
        sep_token=SPECIAL_TOKENS["sep_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        pad_token=SPECIAL_TOKENS["pad_token"],
        pretokenize=args.get("pretokenize", True)
    )
    collate_fn = DynamicPaddingCollator(tokenizer.pad_token_id)
    dataset_lengths = full_dataset.lengths if full_dataset.lengths else None
    use_length_bucketing = bool(args.get("length_bucketing", True) and dataset_lengths)

    # Persist tokenizer once so special tokens survive future reloads.
    tokenizer.save_pretrained(os.path.join(model_directory, "base_model"))

    # Split into train/val for early stopping detection
    total_size = len(full_dataset)
    val_size = max(1, int(total_size * val_split)) if auto_stop and total_size > 1 else 0
    val_size = min(val_size, total_size - 1) if val_size else 0
    train_size = total_size - val_size

    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    eval_val_indices = val_indices[:max_eval_samples] if max_eval_samples and len(val_indices) > max_eval_samples else val_indices

    train_dataset = full_dataset if use_length_bucketing else (Subset(full_dataset, train_indices) if val_size > 0 else full_dataset)
    val_dataset = full_dataset if (use_length_bucketing and eval_val_indices) else (Subset(full_dataset, eval_val_indices) if eval_val_indices else None)
    train_loader = _build_dataloader(
        train_dataset, current_batch_size, True, collate_fn, num_workers, pin_memory,
        lengths=dataset_lengths, indices=train_indices if use_length_bucketing else None,
        use_length_bucketing=use_length_bucketing
    )
    val_loader = _build_dataloader(
        val_dataset, current_batch_size, False, collate_fn, num_workers, pin_memory,
        lengths=dataset_lengths, indices=eval_val_indices if use_length_bucketing else None,
        use_length_bucketing=use_length_bucketing
    ) if eval_val_indices else None

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * max_epochs
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_optimizer_steps = optimizer_steps_per_epoch * max_epochs
    warmup_steps = min(int(args.get("warmup_steps", 100)), max(1, total_optimizer_steps // 5))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)

    # Resume from checkpoint if training_state.pt exists
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    epoch_times = []
    training_state_path = os.path.join(model_directory, "training_state.pt")
    resume_state = None
    if os.path.isfile(training_state_path):
        print("Resuming from training checkpoint...")
        resume_state = torch.load(training_state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        if scaler is not None and 'scaler' in resume_state:
            scaler.load_state_dict(resume_state['scaler'])
        start_epoch = resume_state.get('epoch', 0)
        best_val_loss = float(resume_state.get('best_val_loss', best_val_loss))
        best_epoch = int(resume_state.get('best_epoch', best_epoch))
        epochs_without_improvement = int(resume_state.get('epochs_without_improvement', epochs_without_improvement))
        current_batch_size = int(resume_state.get('current_batch_size', current_batch_size))
        grad_accum_steps = int(resume_state.get('gradient_accumulation_steps', grad_accum_steps))
        if 'python_random_state' in resume_state:
            try:
                random.setstate(resume_state['python_random_state'])
            except Exception:
                print(f"{_C.YELLOW}Warning: Could not restore Python RNG state, re-seeding.{_C.RESET}")
        if 'rng_state' in resume_state:
            try:
                rng = resume_state['rng_state']
                if not isinstance(rng, torch.ByteTensor):
                    rng = rng.byte() if hasattr(rng, 'byte') else torch.ByteTensor(rng)
                torch.set_rng_state(rng)
            except Exception:
                print(f"{_C.YELLOW}Warning: Could not restore CPU RNG state, re-seeding.{_C.RESET}")
        if 'cuda_rng_state' in resume_state and torch.cuda.is_available():
            try:
                cuda_rng = resume_state['cuda_rng_state']
                if not isinstance(cuda_rng, torch.ByteTensor):
                    cuda_rng = cuda_rng.byte() if hasattr(cuda_rng, 'byte') else torch.ByteTensor(cuda_rng)
                torch.cuda.set_rng_state(cuda_rng)
            except Exception:
                print(f"{_C.YELLOW}Warning: Could not restore CUDA RNG state, re-seeding.{_C.RESET}")
        print(f"Resumed from epoch {start_epoch}")
        if current_batch_size != args["batch_size"]:
            train_loader = _build_dataloader(
                train_dataset, current_batch_size, True, collate_fn, num_workers, pin_memory,
                lengths=dataset_lengths, indices=train_indices if use_length_bucketing else None,
                use_length_bucketing=use_length_bucketing
            )
            val_loader = _build_dataloader(
                val_dataset, current_batch_size, False, collate_fn, num_workers, pin_memory,
                lengths=dataset_lengths, indices=eval_val_indices if use_length_bucketing else None,
                use_length_bucketing=use_length_bucketing
            ) if eval_val_indices else None
            total_steps = len(train_loader) * max_epochs

    run_id = resume_state.get("run_id") if resume_state and resume_state.get("run_id") else _make_run_id()
    log_paths = _prepare_training_log_paths(
        model_directory,
        run_id,
        existing_log_dir=resume_state.get("log_dir") if resume_state else None,
    ) if write_diagnostics else {}
    sample_source_indices = eval_val_indices if eval_val_indices else train_indices
    sample_split = "validation" if eval_val_indices else "train"
    sample_pairs = _select_sample_pairs(
        encoded_data,
        sample_source_indices,
        tokenizer,
        sample_preview_count,
        sample_split,
    )
    data_previews = [_truncate_text(row, 600) for row in encoded_data[:data_preview_count]]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())

    if write_diagnostics:
        _write_json(
            log_paths["metadata"],
            {
                "run_id": run_id,
                "status": "running",
                "created_at": _now_iso(),
                "model_directory": model_directory,
                "csv_path": csv_path,
                "base_model_name": BASE_MODEL_NAME,
                "environment": _get_environment_diagnostics(device),
                "args_requested": user_args,
                "args_effective": args,
                "auto_stop": auto_stop,
                "max_epochs": max_epochs,
                "patience": patience,
                "dataset": {
                    "total_rows": total_size,
                    "train_rows": train_size,
                    "validation_rows": val_size,
                    "validation_rows_per_epoch": len(eval_val_indices),
                    "length_stats": length_stats,
                    "truncation_filter": truncation_filter_stats,
                    "previews": data_previews,
                },
                "optimizer": {
                    "type": type(optimizer).__name__,
                    "learning_rate": args.get("learning_rate", 2e-4),
                    "warmup_steps": warmup_steps,
                    "micro_batch_size": current_batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "effective_batch_size": current_batch_size * grad_accum_steps,
                },
                "model": {
                    "trainable_parameters": trainable_param_count,
                    "total_parameters": total_param_count,
                    "trainable_fraction": round(trainable_param_count / max(1, total_param_count), 6),
                    "gradient_checkpointing": use_gradient_checkpointing,
                    "length_bucketing": use_length_bucketing,
                    "pretokenize": args.get("pretokenize", True),
                },
                "sample_pairs": sample_pairs,
                "resume_state_loaded": bool(resume_state),
            },
        )
        _write_json(
            log_paths["summary"],
            {
                "run_id": run_id,
                "status": "running",
                "updated_at": _now_iso(),
                "model_directory": model_directory,
                "current_epoch": start_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                "micro_batch_size": current_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "effective_batch_size": current_batch_size * grad_accum_steps,
                "log_dir": log_paths["run_dir"],
            },
        )
        _append_jsonl(
            log_paths["events"],
            {
                "timestamp": _now_iso(),
                "event": "resume_loaded" if resume_state else "training_started",
                "epoch": start_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "micro_batch_size": current_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "effective_batch_size": current_batch_size * grad_accum_steps,
                "steps_per_epoch": len(train_loader),
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "warmup_steps": warmup_steps,
                "log_dir": log_paths["run_dir"],
            },
        )

    # Print setup info
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    first_sample = full_dataset[0]
    print("First training example (with special tokens):")
    print(decode_data(tokenizer, first_sample['input_ids'], skip_special_tokens=False))
    print(f"\nTotal trainable parameters: {trainable_param_count:,}")
    print(f"{_C.CYAN}Training samples: {train_size}, Validation samples: {val_size}{_C.RESET}")
    if val_loader is not None and len(eval_val_indices) != val_size:
        print(f"{_C.DIM}Validation uses {len(eval_val_indices)} sampled examples per epoch.{_C.RESET}")
    if auto_stop:
        print(f"{_C.CYAN}Mode: AUTO-STOP (patience={patience}, max={max_epochs} epochs){_C.RESET}")
    else:
        print(f"{_C.CYAN}Mode: FIXED ({max_epochs} epochs){_C.RESET}")
    print(f"{_C.DIM}Steps per epoch: {len(train_loader)}, Optimizer steps/epoch: {optimizer_steps_per_epoch}, Warmup: {warmup_steps}{_C.RESET}")
    print(
        f"Logging every {log_every} step(s), num_workers={num_workers}, "
        f"pretokenize={args.get('pretokenize', True)}, length_bucketing={use_length_bucketing}"
    )
    print(f"{_C.DIM}Micro-batch: {current_batch_size}, Grad accumulation: {grad_accum_steps}, Effective batch: {current_batch_size * grad_accum_steps}{_C.RESET}")
    if write_diagnostics:
        print(f"{_C.DIM}Diagnostics logs: {log_paths['run_dir']}{_C.RESET}")

    _emit("training_start", model_dir=model_directory, dataset_size=train_size,
          batch_size=current_batch_size * grad_accum_steps, max_epochs=max_epochs, auto_stop=auto_stop)

    # Suppress scheduler ordering warning (harmless on first step with resume)
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

    training_start = time.time()
    model.train()
    epoch = start_epoch
    try:
        while epoch < max_epochs:
            epoch_loss = 0.0
            epoch_start = time.time()
            steps_processed = 0
            restart_epoch = False
            epoch_tokens = 0
            epoch_padded_tokens = 0
            consecutive_nan = 0
            MAX_CONSECUTIVE_NAN = 50
            epoch_examples = 0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, 1):
                input_ids_cpu = batch['input_ids']
                attention_mask_cpu = batch['attention_mask']
                labels_cpu = batch['labels']
                batch_tokens = int(attention_mask_cpu.sum().item())
                batch_padded_tokens = int(attention_mask_cpu.numel())
                batch_size_actual = int(input_ids_cpu.size(0))
                batch_seq_len_max = int(input_ids_cpu.size(1))
                batch_seq_len_mean = float(attention_mask_cpu.sum(dim=1).float().mean().item())
                input_ids_batch = input_ids_cpu.to(device, non_blocking=pin_memory)
                attention_mask_batch = attention_mask_cpu.to(device, non_blocking=pin_memory)
                labels_batch = labels_cpu.to(device, non_blocking=pin_memory)

                try:
                    if use_amp:
                        with torch.amp.autocast(device.type):
                            outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                            raw_loss = outputs.loss
                            scaled_loss = raw_loss / grad_accum_steps
                    else:
                        outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                        raw_loss = outputs.loss
                        scaled_loss = raw_loss / grad_accum_steps

                    # Skip backward + optimizer step entirely if loss is NaN/Inf
                    if not torch.isfinite(raw_loss):
                        optimizer.zero_grad(set_to_none=True)
                        # Continue to logging below — raw_loss_value will be nan, handled there
                    else:
                        if use_amp:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                        if step % grad_accum_steps == 0 or step == len(train_loader):
                            if use_amp:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                                optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                except torch.cuda.OutOfMemoryError:
                    old_batch_size = current_batch_size
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    grad_accum_steps = max(1, math.ceil(target_effective_batch_size / current_batch_size))
                    print(
                        f"\n  [OOM] Out of memory! Reducing micro_batch to {current_batch_size} "
                        f"and using grad_accum {grad_accum_steps}"
                    )
                    if write_diagnostics:
                        oom_record = {
                            "timestamp": _now_iso(),
                            "event": "cuda_oom",
                            "epoch": epoch + 1,
                            "step": step,
                            "old_micro_batch_size": old_batch_size,
                            "new_micro_batch_size": current_batch_size,
                            "gradient_accumulation_steps": grad_accum_steps,
                            "effective_batch_size": current_batch_size * grad_accum_steps,
                        }
                        oom_record.update(_get_cuda_runtime_stats())
                        _append_jsonl(log_paths["events"], oom_record)
                    train_loader = _build_dataloader(
                        train_dataset, current_batch_size, True, collate_fn, num_workers, pin_memory,
                        lengths=dataset_lengths, indices=train_indices if use_length_bucketing else None,
                        use_length_bucketing=use_length_bucketing
                    )
                    if val_loader is not None:
                        val_loader = _build_dataloader(
                            val_dataset, current_batch_size, False, collate_fn, num_workers, pin_memory,
                            lengths=dataset_lengths, indices=eval_val_indices if use_length_bucketing else None,
                            use_length_bucketing=use_length_bucketing
                        )
                    optimizer.zero_grad(set_to_none=True)
                    restart_epoch = True
                    break

                steps_processed += 1
                raw_loss_value = float(raw_loss.detach().item())
                if math.isfinite(raw_loss_value):
                    epoch_loss += raw_loss_value
                    consecutive_nan = 0
                else:
                    consecutive_nan += 1
                    print(f"{_C.YELLOW}Warning: Non-finite loss ({raw_loss_value}) at step {step}, skipping from average.{_C.RESET}")
                    if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                        print(f"{_C.RED}{_C.BOLD}\n  ** ABORTING EPOCH: {consecutive_nan} consecutive NaN losses — model weights may be corrupted.{_C.RESET}")
                        print(f"  ** Restoring best checkpoint and stopping training.")
                        best_adapter_dir = os.path.join(model_directory, "best_lora_adapter")
                        if os.path.isdir(best_adapter_dir):
                            import shutil
                            adapter_dir = os.path.join(model_directory, "lora_adapter")
                            if os.path.isdir(adapter_dir):
                                shutil.rmtree(adapter_dir)
                            shutil.copytree(best_adapter_dir, adapter_dir)
                            print(f"{_C.GREEN}  ** Restored best_lora_adapter -> lora_adapter{_C.RESET}")
                        raise KeyboardInterrupt("NaN abort")
                avg_loss = epoch_loss / steps_processed
                epoch_tokens += batch_tokens
                epoch_padded_tokens += batch_padded_tokens
                epoch_examples += batch_size_actual

                current_lr = scheduler.get_last_lr()[0]
                if step == 1 or step % log_every == 0 or step == len(train_loader):
                    __print_training_progress__(
                        epoch, max_epochs, step, len(train_loader),
                        raw_loss_value, avg_loss, training_start, total_steps, lr=current_lr
                    )
                    if write_diagnostics:
                        elapsed_time = time.time() - training_start
                        steps_in_epoch = len(train_loader)
                        steps_completed = epoch * steps_in_epoch + step
                        avg_time_per_step = elapsed_time / steps_completed if steps_completed else 0.0
                        eta_seconds = avg_time_per_step * max(0, total_steps - steps_completed)
                        running_duration = max(time.time() - epoch_start, 1e-6)
                        step_record = {
                            "timestamp": _now_iso(),
                            "epoch": epoch + 1,
                            "step": step,
                            "steps_in_epoch": steps_in_epoch,
                            "loss": raw_loss_value,
                            "avg_loss": avg_loss,
                            "learning_rate": current_lr,
                            "elapsed_seconds": round(elapsed_time, 3),
                            "eta_seconds": round(eta_seconds, 3),
                            "micro_batch_size": current_batch_size,
                            "gradient_accumulation_steps": grad_accum_steps,
                            "effective_batch_size": current_batch_size * grad_accum_steps,
                            "batch_examples": batch_size_actual,
                            "batch_tokens": batch_tokens,
                            "batch_padded_tokens": batch_padded_tokens,
                            "batch_padding_ratio": round(1.0 - (batch_tokens / max(1, batch_padded_tokens)), 6),
                            "batch_seq_len_max": batch_seq_len_max,
                            "batch_seq_len_mean": round(batch_seq_len_mean, 3),
                            "epoch_tokens_per_second": round(epoch_tokens / running_duration, 3),
                        }
                        step_record.update(_get_cuda_runtime_stats())
                        _append_jsonl(log_paths["steps"], step_record)

                if step % args["save_every"] == 0:
                    _save_checkpoint(model, optimizer, scheduler, scaler, model_directory, epoch + 1,
                                     best_val_loss=best_val_loss, best_epoch=best_epoch,
                                     epochs_without_improvement=epochs_without_improvement,
                                     current_batch_size=current_batch_size,
                                     gradient_accumulation_steps=grad_accum_steps,
                                     save_training_state=False,
                                     run_id=run_id,
                                     log_dir=log_paths.get("run_dir") if write_diagnostics else None)
                    _emit("checkpoint", epoch=epoch + 1)
                    _sample_after_save(model, tokenizer, encoded_data, device, args, epoch + 1, _emit)
                    if write_diagnostics:
                        _append_jsonl(
                            log_paths["events"],
                            {
                                "timestamp": _now_iso(),
                                "event": "checkpoint_saved",
                                "epoch": epoch + 1,
                                "step": step,
                                "avg_loss": avg_loss,
                                "micro_batch_size": current_batch_size,
                                "gradient_accumulation_steps": grad_accum_steps,
                            },
                        )
                    if args.get("enableSampleMode", False):
                        sample_prompts = ["Hello, how are you?", "What's your name?", "Tell me a joke."]
                        sample_prompt = random.choice(sample_prompts)
                        response = generate_responses(model, tokenizer, sample_prompt, device=device, args=args, clean_result=True)
                        response_clean = response.strip()[:200]
                        print(f"{_C.MAGENTA}  Sample — Prompt: {sample_prompt} | Response: {response_clean}{_C.RESET}")
                        _emit("sample", epoch=epoch+1, step=step, prompt=sample_prompt, response=response[:1500])
                        if write_diagnostics:
                            _write_sample_records(
                                log_paths,
                                [{
                                    "timestamp": _now_iso(),
                                    "epoch": epoch + 1,
                                    "split": "manual",
                                    "source_index": -1,
                                    "prompt": _truncate_text(sample_prompt, 500),
                                    "target": "",
                                    "generated": _truncate_text(response, 800),
                                    "sample_max_new_tokens": args.get("max_new_tokens", 256),
                                }],
                            )
                        model.train()

            if restart_epoch:
                print(f"{_C.RED}{_C.BOLD}  [OOM] Restarting epoch {epoch+1} with smaller micro_batch.{_C.RESET}")
                if write_diagnostics:
                    _append_jsonl(
                        log_paths["events"],
                        {
                            "timestamp": _now_iso(),
                            "event": "epoch_restarted_after_oom",
                            "epoch": epoch + 1,
                            "micro_batch_size": current_batch_size,
                            "gradient_accumulation_steps": grad_accum_steps,
                        },
                    )
                continue

            epoch_duration = time.time() - epoch_start
            epoch_times.append(epoch_duration)
            num_steps = max(1, steps_processed)
            avg_train_loss = epoch_loss / num_steps if num_steps > 0 else 0.0

            if val_loader is not None:
                val_loss = _evaluate(model, val_loader, device, use_amp, non_blocking=pin_memory)
            else:
                val_loss = avg_train_loss

            epoch_padding_ratio = 1.0 - (epoch_tokens / max(1, epoch_padded_tokens))
            epoch_tokens_per_second = epoch_tokens / max(epoch_duration, 1e-6)
            epoch_examples_per_second = epoch_examples / max(epoch_duration, 1e-6)

            print(f"{_C.BLUE}{_C.BOLD}\n--- Epoch {epoch+1} Summary ---{_C.RESET}")
            print(f"{_C.GREEN}  Train Loss: {avg_train_loss:.4f}{_C.RESET}")
            if val_loader is not None:
                print(f"{_C.GREEN}  Val Loss:   {val_loss:.4f}{_C.RESET}")
            print(f"{_C.DIM}  Duration:   {_format_time(epoch_duration)}{_C.RESET}")
            print(f"{_C.DIM}  Throughput: {epoch_tokens_per_second:.1f} tok/s, {epoch_examples_per_second:.2f} samples/s{_C.RESET}")
            print(f"{_C.DIM}  Padding:    {epoch_padding_ratio * 100:.1f}%{_C.RESET}")

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            if auto_stop:
                estimated_remaining = patience - epochs_without_improvement
                remaining_time = avg_epoch_time * estimated_remaining
                print(f"{_C.DIM}  Est. time if no improvement: {_format_time(remaining_time)} ({estimated_remaining} epochs){_C.RESET}")
            else:
                remaining_epochs = max_epochs - (epoch + 1)
                remaining_time = avg_epoch_time * remaining_epochs
                print(f"{_C.DIM}  Est. remaining: {_format_time(remaining_time)} ({remaining_epochs} epochs left){_C.RESET}")

            is_new_best = False
            if auto_stop:
                if val_loss < (best_val_loss - min_improvement):
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    is_new_best = True
                    print(f"{_C.GREEN}{_C.BOLD}  >> New best! (val loss: {best_val_loss:.4f}){_C.RESET}")
                    _save_best_checkpoint(model, model_directory)
                    if write_diagnostics:
                        _append_jsonl(
                            log_paths["events"],
                            {
                                "timestamp": _now_iso(),
                                "event": "new_best_checkpoint",
                                "epoch": epoch + 1,
                                "train_loss": avg_train_loss,
                                "val_loss": val_loss,
                            },
                        )
                else:
                    epochs_without_improvement += 1
                    print(f"{_C.YELLOW}  >> No improvement for {epochs_without_improvement}/{patience} epochs (best: {best_val_loss:.4f} at epoch {best_epoch}){_C.RESET}")

            if write_diagnostics:
                epoch_record = {
                    "timestamp": _now_iso(),
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "duration_seconds": round(epoch_duration, 3),
                    "duration_human": _format_time(epoch_duration),
                    "micro_batch_size": current_batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "effective_batch_size": current_batch_size * grad_accum_steps,
                    "tokens_processed": epoch_tokens,
                    "padded_tokens_processed": epoch_padded_tokens,
                    "padding_ratio": round(epoch_padding_ratio, 6),
                    "tokens_per_second": round(epoch_tokens_per_second, 3),
                    "samples_per_second": round(epoch_examples_per_second, 3),
                    "best_epoch": best_epoch,
                    "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                    "is_new_best": is_new_best,
                    "epochs_without_improvement": epochs_without_improvement,
                }
                epoch_record.update(_get_cuda_runtime_stats())
                _append_jsonl(log_paths["epochs"], epoch_record)

            if write_diagnostics and sample_pairs and sample_log_every_epochs > 0 and ((epoch + 1) % sample_log_every_epochs == 0):
                try:
                    _generate_diagnostic_samples(
                        model,
                        tokenizer,
                        sample_pairs,
                        device,
                        args,
                        epoch + 1,
                        log_paths,
                    )
                    _append_jsonl(
                        log_paths["events"],
                        {
                            "timestamp": _now_iso(),
                            "event": "diagnostic_samples_written",
                            "epoch": epoch + 1,
                            "sample_count": len(sample_pairs),
                        },
                    )
                except Exception as e:
                    print(f"{_C.YELLOW}Warning: Diagnostic sample generation failed: {e}{_C.RESET}")
                    _append_jsonl(
                        log_paths["events"],
                        {
                            "timestamp": _now_iso(),
                            "event": "diagnostic_samples_failed",
                            "epoch": epoch + 1,
                            "sample_count": len(sample_pairs),
                            "error": str(e),
                        },
                    )

            _emit("epoch_done", epoch=epoch+1, max_epochs=max_epochs, train_loss=avg_train_loss,
                  val_loss=val_loss if val_loader is not None else None,
                  duration=_format_time(epoch_duration), best=is_new_best)

            _sample_after_save(model, tokenizer, encoded_data, device, args, epoch + 1, _emit)
            _save_checkpoint(model, optimizer, scheduler, scaler, model_directory, epoch + 1,
                             best_val_loss=best_val_loss, best_epoch=best_epoch,
                             epochs_without_improvement=epochs_without_improvement,
                             current_batch_size=current_batch_size,
                             gradient_accumulation_steps=grad_accum_steps,
                             save_training_state=True,
                             run_id=run_id,
                             log_dir=log_paths.get("run_dir") if write_diagnostics else None)
            _emit("checkpoint", epoch=epoch + 1)

            if write_diagnostics:
                _write_json(
                    log_paths["summary"],
                    {
                        "run_id": run_id,
                        "status": "running",
                        "updated_at": _now_iso(),
                        "model_directory": model_directory,
                        "current_epoch": epoch + 1,
                        "best_epoch": best_epoch,
                        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                        "last_train_loss": avg_train_loss,
                        "last_val_loss": val_loss,
                        "micro_batch_size": current_batch_size,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "effective_batch_size": current_batch_size * grad_accum_steps,
                        "log_dir": log_paths["run_dir"],
                    },
                )

            if auto_stop and epochs_without_improvement >= patience:
                print(f"{_C.YELLOW}{_C.BOLD}\n** Auto-stopped: no improvement for {patience} epochs. Best was epoch {best_epoch} (val loss: {best_val_loss:.4f}){_C.RESET}")
                print(f"{_C.GREEN}** Your best checkpoint is already saved.{_C.RESET}")
                total_training_time = time.time() - training_start
                if write_diagnostics:
                    _append_jsonl(
                        log_paths["events"],
                        {
                            "timestamp": _now_iso(),
                            "event": "training_auto_stopped",
                            "epoch": epoch + 1,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "total_time_seconds": round(total_training_time, 3),
                        },
                    )
                    _write_json(
                        log_paths["summary"],
                        {
                            "run_id": run_id,
                            "status": "auto_stopped",
                            "updated_at": _now_iso(),
                            "model_directory": model_directory,
                            "completed_epoch": epoch + 1,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "total_time_seconds": round(total_training_time, 3),
                            "total_time_human": _format_time(total_training_time),
                            "micro_batch_size": current_batch_size,
                            "gradient_accumulation_steps": grad_accum_steps,
                            "effective_batch_size": current_batch_size * grad_accum_steps,
                            "log_dir": log_paths["run_dir"],
                        },
                    )
                _emit("training_done", total_time=_format_time(total_training_time),
                      best_epoch=best_epoch, best_val_loss=best_val_loss, stopped_early=True)
                break

            epoch += 1

        if not (auto_stop and epochs_without_improvement >= patience):
            total_training_time = time.time() - training_start
            print(f"{_C.GREEN}{_C.BOLD}\nTraining completed in {_format_time(total_training_time)}.{_C.RESET}")
            if write_diagnostics:
                _append_jsonl(
                    log_paths["events"],
                    {
                        "timestamp": _now_iso(),
                        "event": "training_completed",
                        "completed_epoch": epoch if epoch == max_epochs else epoch + 1,
                        "best_epoch": best_epoch if auto_stop else None,
                        "best_val_loss": best_val_loss if auto_stop and best_val_loss < float('inf') else None,
                        "total_time_seconds": round(total_training_time, 3),
                    },
                )
                _write_json(
                    log_paths["summary"],
                    {
                        "run_id": run_id,
                        "status": "completed",
                        "updated_at": _now_iso(),
                        "model_directory": model_directory,
                        "completed_epoch": epoch if epoch == max_epochs else epoch + 1,
                        "best_epoch": best_epoch if auto_stop else None,
                        "best_val_loss": best_val_loss if auto_stop and best_val_loss < float('inf') else None,
                        "total_time_seconds": round(total_training_time, 3),
                        "total_time_human": _format_time(total_training_time),
                        "micro_batch_size": current_batch_size,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "effective_batch_size": current_batch_size * grad_accum_steps,
                        "log_dir": log_paths["run_dir"],
                    },
                )
            _emit("training_done", total_time=_format_time(total_training_time),
                  best_epoch=best_epoch if auto_stop else None,
                  best_val_loss=best_val_loss if auto_stop and best_val_loss < float('inf') else None)
    except Exception as exc:
        if write_diagnostics:
            failure_record = {
                "timestamp": _now_iso(),
                "event": "training_failed",
                "epoch": epoch + 1,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "micro_batch_size": current_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
            }
            failure_record.update(_get_cuda_runtime_stats())
            _append_jsonl(log_paths["events"], failure_record)
            _write_json(
                log_paths["summary"],
                {
                    "run_id": run_id,
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "model_directory": model_directory,
                    "epoch": epoch + 1,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "best_epoch": best_epoch,
                    "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                    "micro_batch_size": current_batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "effective_batch_size": current_batch_size * grad_accum_steps,
                    "log_dir": log_paths["run_dir"],
                },
            )
        raise


def _save_best_checkpoint(model, model_directory: str) -> None:
    best_adapter_dir = os.path.join(model_directory, "best_lora_adapter")
    model.save_pretrained(best_adapter_dir)
    print(f"{_C.GREEN}Best checkpoint updated.{_C.RESET}")


def _save_checkpoint(model, optimizer, scheduler, scaler, model_directory: str,
                     epoch: int, best_val_loss: float = None, best_epoch: int = None,
                     epochs_without_improvement: int = 0,
                     current_batch_size: Optional[int] = None,
                     gradient_accumulation_steps: Optional[int] = None,
                     save_training_state: bool = True,
                     run_id: Optional[str] = None,
                     log_dir: Optional[str] = None) -> None:
    """Saves LoRA adapter plus optimizer/scheduler/RNG state for resume."""
    adapter_dir = os.path.join(model_directory, "lora_adapter")
    training_state_path = os.path.join(model_directory, "training_state.pt")

    model.save_pretrained(adapter_dir)

    if save_training_state:
        state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
            'current_batch_size': current_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'run_id': run_id,
            'log_dir': log_dir,
            'python_random_state': random.getstate(),
            'rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda_rng_state'] = torch.cuda.get_rng_state()
        if scaler is not None:
            state['scaler'] = scaler.state_dict()
        torch.save(state, training_state_path)

    checkpoint_kind = "resume state + adapter" if save_training_state else "adapter only"
    print(f"{_C.GREEN}Checkpoint saved (epoch {epoch}, {checkpoint_kind}){_C.RESET}")


def _sample_after_save(model, tokenizer, encoded_data, device, args, epoch, _emit=None):
    """Generate a response from a random training prompt."""
    original_use_cache = None
    was_training = bool(getattr(model, "training", False))
    try:
        # Pick a random training example and extract the prompt (context before sep)
        raw = random.choice(encoded_data)
        sep = tokenizer.sep_token or SPECIAL_TOKENS["sep_token"]
        bos = tokenizer.bos_token or SPECIAL_TOKENS["bos_token"]
        if sep in raw:
            prompt = raw.split(sep)[0].replace(bos, "").strip()
        else:
            prompt = raw.replace(bos, "").strip()[:80]

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            original_use_cache = model.config.use_cache
            model.config.use_cache = True

        sample_args = dict(args)
        sample_args["max_new_tokens"] = min(96, int(args.get("max_new_tokens", 256)))

        generated = generate_responses(
            model, tokenizer, prompt,
            device=device, args=sample_args, clean_result=True,
        )

        # Find the expected reply from training data
        expected_raw = raw.split(sep)[1].strip() if sep in raw else "?"
        # Clean special tokens from expected
        for tok in (tokenizer.eos_token, tokenizer.bos_token, tokenizer.pad_token):
            if tok:
                expected_raw = expected_raw.replace(tok, "")
        expected = expected_raw.strip()[:200]

        # Truncate model output and collapse to single line for readability
        generated_clean = generated.strip()[:200]

        # Clean print format
        print(f"{_C.MAGENTA}\n{'='*50}{_C.RESET}")
        print(f"{_C.MAGENTA}  Sample after epoch {epoch}{_C.RESET}")
        print(f"{_C.MAGENTA}  Prompt:   {prompt[:120]}{_C.RESET}")
        print(f"{_C.MAGENTA}  Expected: {expected[:120]}{_C.RESET}")
        print(f"{_C.MAGENTA}  Model:    {generated_clean[:120]}{_C.RESET}")
        print(f"{'='*50}\n")

        if _emit is not None:
            _emit("sample_after_epoch", epoch=epoch, prompt=prompt, expected=expected, generated=generated_clean)
    except Exception as e:
        print(f"{_C.YELLOW}Warning: Post-save sample failed: {e}{_C.RESET}")
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(was_training)


# --- Clean Text ---

def clean_text(uncleaned_text: str, pad_token: str = "", sep_token: str = "",
               eos_token: str = "", bos_token: str = "") -> str:
    import re as _re
    special_tokens_dict = {
        'pad_token': pad_token,
        'sep_token': sep_token,
        'eos_token': eos_token,
        'bos_token': bos_token
    }
    before_sep, sep, after_sep = uncleaned_text.partition(sep_token)
    after_sep = after_sep.replace(bos_token, '').strip()
    while (sep_token and after_sep.startswith(sep_token)) or (bos_token and after_sep.startswith(bos_token)):
        if after_sep.startswith(sep_token):
            after_sep = after_sep[len(sep_token):].strip()
        if after_sep.startswith(bos_token):
            after_sep = after_sep[len(bos_token):].strip()
    split_text = after_sep.split(sep_token)[0]
    for token in special_tokens_dict.values():
        if token:
            split_text = split_text.replace(token, '').strip()

    # Strip any Phi-3.5 native tokens that leak through
    for tok in ("<|end|>", "<|endoftext|>", "<|user|>", "<|assistant|>",
                "<|system|>", "<|placeholder1|>", "<|placeholder2|>",
                "<|placeholder3|>", "<|placeholder4|>"):
        split_text = split_text.split(tok)[0]

    # Strip mojibake / replacement chars
    split_text = split_text.replace("\ufffd", "").replace("\u00ef\u00bf\u00bd", "").strip()

    # Convert [NL] tokens back to actual newlines
    split_text = split_text.replace("[NL]", "\n").replace("[Nl]", "\n").replace("[nl]", "\n")

    # Keep Discord emoji codes and placeholders — they're part of the personality

    # Remove non-Latin garbage (multilingual hallucinations)
    split_text = _re.sub(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]+', '', split_text)  # Cyrillic
    split_text = _re.sub(r'[\u3000-\u9FFF\uF900-\uFAFF]+', '', split_text)  # CJK
    split_text = _re.sub(r'[\u0100-\u024F]{3,}', '', split_text)  # Long Latin Extended runs (Hungarian/Polish gibberish)

    # Remove LaTeX-like artifacts (but not plain dollar amounts)
    split_text = _re.sub(r'\$[a-zA-Z\\{][^\s]*', '', split_text)
    split_text = _re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', split_text)

    # Clean up lines
    lines = split_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Strip trailing symbol garbage without clipping real word endings.
        line = _re.sub(r'\s*[~^`|\\=\$%]{1,3}$', '', line.rstrip())
        # Only keep lines with actual content
        if line.strip() and _re.search(r'[a-zA-Z0-9]', line):
            cleaned_lines.append(line.strip())
    split_text = '\n'.join(cleaned_lines)

    # Remove truncated words at the end (cut off mid-syllable)
    _ok_short = {'i', 'a', 'u', 'r', 'w', 'ok', 'no', 'ya', 'ye', 'yo', 'hi', 'ur',
                 'im', 'my', 'me', 'we', 'he', 'it', 'so', 'do', 'go', 'be', 'up',
                 'an', 'am', 'as', 'at', 'by', 'if', 'in', 'is', 'of', 'on', 'or',
                 'to', 'us', 'lol', 'omg', 'wtf', 'idk', 'ngl', 'smh', 'rip', 'gg',
                 'ez', 'fr', 'fs', 'af', 'np', 'ty', 'gn', 'gm', 'ew', 'uh', 'hm',
                 'oh', 'ah', 'ow', 'the', 'and', 'but', 'for', 'not', 'you', 'all',
                 'can', 'was', 'one', 'out', 'got', 'has', 'how', 'its', 'now', 'old',
                 'see', 'who', 'did', 'get', 'say', 'too', 'use', 'bro', 'tho', 'rn',
                 'bc', 'nah', 'yea', 'yes', 'tbh', 'xd', '3'}
    if split_text.strip():
        words = split_text.rstrip().split()
        if words:
            last = words[-1].rstrip('.,!?').lower()
            if last not in _ok_short and len(last) <= 3 and last.isalpha():
                split_text = ' '.join(words[:-1])

    # Model sometimes chains fragments with pipes
    split_text = _re.split(r'\s+\|\s+', split_text, maxsplit=1)[0].strip()

    # Keep replies to max 2 lines
    if split_text:
        split_text = '\n'.join(split_text.splitlines()[:2]).strip()
        split_text = _truncate_text(split_text, limit=180)

    return split_text.strip()


# Runtime cleanup override for weak LoRA outputs.
def _clean_text_runtime(uncleaned_text: str, pad_token: str = "", sep_token: str = "",
                        eos_token: str = "", bos_token: str = "") -> str:
    import re as _re

    special_tokens_dict = {
        'pad_token': pad_token,
        'sep_token': sep_token,
        'eos_token': eos_token,
        'bos_token': bos_token
    }
    _, _, after_sep = uncleaned_text.partition(sep_token)
    after_sep = after_sep.replace(bos_token, '').strip()
    while (sep_token and after_sep.startswith(sep_token)) or (bos_token and after_sep.startswith(bos_token)):
        if after_sep.startswith(sep_token):
            after_sep = after_sep[len(sep_token):].strip()
        if after_sep.startswith(bos_token):
            after_sep = after_sep[len(bos_token):].strip()
    split_text = after_sep.split(sep_token)[0]
    for token in special_tokens_dict.values():
        if token:
            split_text = split_text.replace(token, '').strip()

    for tok in ("<|end|>", "<|endoftext|>", "<|user|>", "<|assistant|>",
                "<|system|>", "<|placeholder1|>", "<|placeholder2|>",
                "<|placeholder3|>", "<|placeholder4|>"):
        split_text = split_text.split(tok)[0]

    split_text = split_text.replace("\ufffd", "").replace("ï¿½", "").strip()
    split_text = _re.sub(r"\[\s*NL\s*\]", "\n", split_text, flags=_re.I)
    split_text = _re.sub(r"\bI\s+MAGE\b", "IMAGE", split_text, flags=_re.I)
    split_text = _re.sub(
        r"\[(?:USER|PINGUSER|PING ?ME|IMAGE|I ?MAGE|VIDEO|AUDIO|ATTACHMENT|FILE|PATH|EMAIL|PHONE|IP|CHANNEL|LINK|MEDIA_LINK|DISCORD_LINK|NONE)\]",
        "",
        split_text,
        flags=_re.I,
    )
    split_text = _re.sub(
        r"\[(?:PHOTOS?|PHOTO|IMAGE OF|VIDEO PLAYER READY WITH|SEND IT TO MY DISCORD|PHRASE|PING:? [^\]]+)\]",
        "",
        split_text,
        flags=_re.I,
    )
    split_text = _re.sub(r"https?://\S+", "", split_text, flags=_re.I)

    # Remove non-ASCII garbage (Cyrillic, CJK, hex byte artifacts)
    split_text = _re.sub(r'[\u0400-\u04FF\u0500-\u052F]+', '', split_text)  # Cyrillic
    split_text = _re.sub(r'[\u3000-\u9FFF]+', '', split_text)  # CJK
    split_text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', split_text)  # control chars

    # Remove stray single characters at end of words/lines (model artifacts)
    split_text = _re.sub(r'(?<=[a-zA-Z])[A-Z](?=\s|$)', '', split_text)  # random capital after lowercase
    split_text = _re.sub(r'\s+[a-zA-Z~^`|\\+=/;:]{1}(?:\s|$)', ' ', split_text)  # lone garbage chars

    lines = split_text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = _re.sub(r'\s*[~^`|\\]{1,3}$', '', line.rstrip())
        line = _re.sub(r'(?:\s*[\],;:}{)(]{2,}\s*)+', ' ', line).strip()
        line = _re.sub(r'\[[^\]]{0,40}$', '', line).strip()
        # Remove trailing single non-word char
        line = _re.sub(r'\s*[a-zA-Z]{1}$', '', line.rstrip())
        if line and _re.search(r'[a-zA-Z0-9]', line):
            cleaned_lines.append(line)
    split_text = '\n'.join(cleaned_lines)
    split_text = _re.split(r'\s+\|\s+', split_text, maxsplit=1)[0].strip()

    # Take only first 2 lines max
    split_text = '\n'.join(line.strip() for line in split_text.splitlines() if line.strip())
    if split_text:
        split_text = '\n'.join(split_text.splitlines()[:2]).strip()
        split_text = _truncate_text(split_text, limit=180)
    return split_text.strip()


def _looks_bad_generation(text: str) -> bool:
    import re as _re
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    if "http://" in stripped or "https://" in stripped:
        return True
    if stripped.count("<|") >= 1 or "<[" in stripped:
        return True
    if len(_re.findall(r"\[[^\]]*\]", stripped)) >= 2:
        return True
    if sum(ch in "[]{}|" for ch in stripped) >= 6:
        return True
    allowed_emoji = set("😭😂🥺😔😎😳😤😅🤣❤️💀✨")
    non_ascii = sum(1 for ch in stripped if ord(ch) > 127 and ch not in allowed_emoji and not ch.isspace())
    if non_ascii >= max(6, len(stripped) // 6):
        return True
    return False


# --- Prompt & Generation ---

def format_prompt(prompt_text: str, start_token: str = SPECIAL_TOKENS["bos_token"],
                  sep_token: str = SPECIAL_TOKENS["sep_token"]) -> str:
    return f"{start_token} {prompt_text} {sep_token}"


def _has_lora_adapter(model) -> bool:
    """Check if the model has a LoRA adapter loaded."""
    return isinstance(model, PeftModel)


_INPUT_VOCAB = None

def _build_input_vocab():
    """Build a set of known words from training data for input correction."""
    import re
    global _INPUT_VOCAB
    if _INPUT_VOCAB is not None:
        return _INPUT_VOCAB

    vocab = set()
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")
    if os.path.isfile(csv_path):
        try:
            import csv as _csv
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = _csv.reader(f)
                next(reader, None)
                for row in reader:
                    for field in row:
                        for word in re.split(r'[^a-zA-Z]+', field.lower()):
                            if len(word) > 1:
                                vocab.add(word)
        except Exception:
            pass
    _INPUT_VOCAB = vocab
    return vocab


def _correct_input(text: str) -> str:
    """Auto-correct user input using fuzzy matching against training vocabulary.
    Only corrects words that aren't in the training data or dictionary.
    Fast — uses edit distance 1 only."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return text

    vocab = _build_input_vocab()
    spell = SpellChecker()
    # Add our training vocab as known words
    spell.word_frequency.load_words(vocab)

    words = text.split()
    corrected = []
    for word in words:
        clean = word.lower().strip('.,!?;:\'"()-')
        # Skip short words, numbers, URLs, mentions
        if len(clean) <= 2 or clean.isdigit() or word.startswith(('http', '@', '#', '/')):
            corrected.append(word)
            continue
        # Already known — keep as-is
        if clean in spell or clean in vocab:
            corrected.append(word)
            continue
        # Try to correct
        correction = spell.correction(clean)
        if correction and correction != clean:
            # Preserve original case
            if word[0].isupper():
                correction = correction[0].upper() + correction[1:]
            if word.isupper():
                correction = correction.upper()
            corrected.append(correction)
        else:
            corrected.append(word)
    return ' '.join(corrected)


def generate_responses(model, tokenizer, prompt_text: str,
                       device: Optional[torch.device] = None,
                       args: Optional[Dict[str, Any]] = None, clean_result: bool = False) -> str:
    if args is None:
        args = create_args()
    if device is None:
        device = _get_device()

    # Auto-correct input typos before generation (skip during training)
    if args.get("auto_correct_input", True) and not getattr(model, "training", False):
        prompt_text = _correct_input(prompt_text)

    was_training = bool(getattr(model, "training", False))
    model.eval()

    # Use native Phi-3.5 chat format when no LoRA adapter is loaded,
    # since the base model doesn't understand our custom special tokens
    if _has_lora_adapter(model):
        formatted_prompt = format_prompt(prompt_text, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)
    else:
        formatted_prompt = f"<|system|>\nYou are {BOT_NAME}. You are a real person who talks casually in Discord. You are chill, funny, sometimes chaotic, and you care about your friends. You never say you are a bot or AI.<|end|>\n<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"

    encoded = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Stop generation after too many newlines (messages are short bursts)
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)
    newline_id = newline_token_id[0] if newline_token_id else None

    class NewlineStopCriteria(StoppingCriteria):
        def __init__(self, newline_id, max_newlines=6, start_len=0):
            self.newline_id = newline_id
            self.max_newlines = max_newlines
            self.start_len = start_len
        def __call__(self, input_ids, scores, **kwargs):
            if self.newline_id is None:
                return False
            generated = input_ids[0][self.start_len:]
            return (generated == self.newline_id).sum().item() >= self.max_newlines

    max_newlines = args.get("max_newlines", 2)
    stopping_criteria = StoppingCriteriaList([
        NewlineStopCriteria(newline_id, max_newlines=max_newlines, start_len=input_ids.shape[1])
    ])

    do_sample = bool(args.get("do_sample", True))
    temperature = float(args.get("temperature", 0.7))
    if temperature <= 0.0:
        do_sample = False
        temperature = 1.0

    generate_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": args.get("max_new_tokens", 256),
        "temperature": temperature,
        "top_k": args["top_k"],
        "top_p": args["top_p"],
        "repetition_penalty": args["repetition_penalty"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": (
            tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"])
            if _has_lora_adapter(model)
            else tokenizer.convert_tokens_to_ids("<|end|>")
        ),
        "num_return_sequences": 1,
        "stopping_criteria": stopping_criteria,
        "use_cache": True,
    }

    try:
        with torch.no_grad():
            try:
                output = model.generate(input_ids, **generate_kwargs)
            except AttributeError as exc:
                # Phi-3.5 + some Transformers/PEFT combinations can trip over DynamicCache.seen_tokens.
                if "DynamicCache" not in str(exc) and "seen_tokens" not in str(exc):
                    raise
                generate_kwargs["use_cache"] = False
                output = model.generate(input_ids, **generate_kwargs)
    finally:
        model.train(was_training)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    if clean_result:
        if _has_lora_adapter(model):
            generated_text = _clean_text_runtime(
                generated_text,
                pad_token=tokenizer.pad_token,
                sep_token=tokenizer.sep_token,
                eos_token=tokenizer.eos_token,
                bos_token=tokenizer.bos_token
            )
            if _looks_bad_generation(generated_text):
                retry_kwargs = dict(generate_kwargs)
                retry_kwargs.update({
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "repetition_penalty": min(1.12, float(args.get("repetition_penalty", 1.2))),
                    "max_new_tokens": min(48, int(args.get("max_new_tokens", 256))),
                })
                with torch.no_grad():
                    retry_output = model.generate(input_ids, **retry_kwargs)
                retry_text = tokenizer.decode(retry_output[0], skip_special_tokens=False)
                retry_text = _clean_text_runtime(
                    retry_text,
                    pad_token=tokenizer.pad_token,
                    sep_token=tokenizer.sep_token,
                    eos_token=tokenizer.eos_token,
                    bos_token=tokenizer.bos_token
                )
                if retry_text and not _looks_bad_generation(retry_text):
                    generated_text = retry_text
        else:
            marker = "<|assistant|>"
            if marker in generated_text:
                generated_text = generated_text.split(marker, 1)[1]
            for tok in ("<|end|>", "<|endoftext|>", "<|user|>", "<|system|>"):
                generated_text = generated_text.split(tok)[0]
            generated_text = generated_text.strip()
    return _apply_bot_name(generated_text)
