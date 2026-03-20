# SimpleGPT - Discord Chatbot Fine-Tuning

Fine-tune Microsoft Phi-2 (2.7B) with LoRA to generate conversational responses trained on your Discord messages.

## Requirements
- Python 3.8+
- NVIDIA GPU recommended (works on CPU but much slower)
- Packages: `torch transformers peft pandas requests`

## Setup

```bash
git clone https://github.com/realdarter/SimpleGPT
cd SimpleGPT
pip install torch transformers peft pandas requests
```

Preferably install PyTorch with CUDA support: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

## Prepare Dataset

Create a CSV file with `context` and `reply` columns:

```csv
context,reply
How are you doing so far?,I am doing fine!
lol!,Thats funny
"I need to talk to you","What, do, you, want"
```

Place it at `checkpoint/run3/cleaned.csv` (or adjust the path in `train.py`).

## Train

```python
import os
from chat_gen import create_args, train_model

model_directory = 'checkpoint/run3'
csv_path = os.path.join(model_directory, 'cleaned.csv')

args = create_args(
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    save_every=500,
    max_length=512,
)

train_model(model_directory, csv_path, args)
```

Or just run `python train.py`.

Phi-2 (~5GB) downloads automatically on first run. The LoRA adapter (~50MB) is saved separately.

## Test

```python
from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer,
    ensure_tokens, SPECIAL_TOKENS, _get_device
)

model_directory = 'checkpoint/run3'
model, tokenizer = load_model_and_tokenizer(model_directory, download=False)
device = _get_device()
ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
model.to(device)

args = create_args(max_length=512, temperature=0.8, top_k=60, top_p=0.92, repetition_penalty=1.2)

prompt_text = input("Input: ")
response = generate_responses(model, tokenizer, prompt_text, device=device, args=args, clean_result=True)
print(f"Generated Response: {response}")
```

Or just run `python test.py`.

## Data Cleaning

Run `python runclean.py` for an interactive CSV cleaner that lets you review and delete bad training pairs.

## Discord Bot

1. Put your Discord token in `token.txt`
2. Update the `channel_id` in `main.py`
3. Run `python main.py`

## Architecture

- **Base model**: Microsoft Phi-2 (2.7B parameters)
- **Fine-tuning**: LoRA (trains ~1% of parameters)
- **Training features**: Mixed precision (FP16), gradient clipping, linear LR warmup + decay, lazy tokenization
- **Directory structure**:
  ```
  checkpoint/run3/
  ├── cleaned.csv              # your training data
  ├── base_model/              # Phi-2 weights (downloaded once)
  ├── lora_adapter/            # your fine-tune (small, portable)
  └── training_state.pt        # optimizer/scheduler state for resuming
  ```
