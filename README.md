# SimpleGPT - Chatbot Fine-Tuning

Fine-tune Microsoft Phi-2 (2.7B) with LoRA to generate conversational responses trained on your messages.

## Requirements
- Python 3.10+
- NVIDIA GPU recommended (works on CPU but much slower)
- Packages: `torch transformers peft pandas`

## Setup

```bash
git clone https://github.com/realdarter/SimpleGPT
cd SimpleGPT
pip install torch transformers peft pandas
```

Preferably install PyTorch with CUDA support: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

Run `python torch-debug.py` to check your environment — it auto-detects your GPU, CUDA version, missing packages, and tells you the exact install command.

## Prepare Dataset

Create a CSV file with `context` and `reply` columns:

```csv
context,reply
How are you doing so far?,I am doing fine!
lol!,Thats funny
"I need to talk to you","What, do, you, want"
```

Place it at `checkpoint/run3/cleaned.csv` (or adjust the path in `train.py`).

## Training Args

`create_args()` controls both training and generation. Here's what each parameter does:

| Arg | Default | What it does |
|-----|---------|-------------|
| `num_epochs` | 0 | **0 = auto-stop** (stops when loss plateaus). Set to a number like 3 to force exactly 3 epochs |
| `patience` | 3 | (Auto-stop only) How many epochs without improvement before stopping |
| `max_epochs` | 50 | (Auto-stop only) Hard cap so it doesn't run forever |
| `val_split` | 0.1 | (Auto-stop only) Fraction of data held out for validation (10%) |
| `batch_size` | 1 | How many examples to process at once. Higher = faster but uses more VRAM |
| `learning_rate` | 2e-4 | How aggressively the model updates. Too high = unstable, too low = slow learning |
| `warmup_steps` | 100 | Steps where learning rate ramps up from 0. Prevents early instability |
| `save_every` | 500 | Save a checkpoint every N steps during training |
| `max_length` | 512 | Max token length for training sequences (context + reply) |
| `max_new_tokens` | 256 | Max tokens the model generates in a response |
| `temperature` | 0.7 | Randomness of output. Lower = more predictable, higher = more creative |
| `top_k` | 50 | Only consider the top K most likely next words |
| `top_p` | 0.95 | Only consider words within this cumulative probability (nucleus sampling) |
| `repetition_penalty` | 1.2 | Penalizes repeating the same words. 1.0 = no penalty |
| `enableSampleMode` | False | If True, generates sample responses during training so you can watch quality |

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

Phi-2 (~5GB) downloads automatically on first run. The LoRA adapter (~50MB) is saved separately. Training automatically resumes from checkpoints if interrupted.

## Test

```bash
python test.py
```

or

```bash
python main.py
```

Both load the model and let you chat in the terminal. Type `quit` to exit (main.py only).

## Data Cleaning

Run `python runclean.py` for an interactive CSV cleaner that lets you review and delete bad training pairs. Progress is tracked per-file so you can pick up where you left off.

## Architecture

- **Base model**: Microsoft Phi-2 (2.7B parameters)
- **Fine-tuning**: LoRA (trains ~1% of parameters)
- **Training features**: Mixed precision (FP16), gradient clipping, linear LR warmup + decay, lazy tokenization, full checkpoint resume
- **Directory structure**:
  ```
  checkpoint/run3/
  |-- cleaned.csv              # your training data
  |-- base_model/              # Phi-2 weights (downloaded once)
  |-- lora_adapter/            # your fine-tune (small, portable)
  |-- training_state.pt        # optimizer/scheduler/RNG state for resuming
  ```

## Files

| File | Purpose |
|------|---------|
| `chat_gen.py` | Core module — dataset, tokenization, training, generation |
| `train.py` | Training entry point |
| `test.py` | Interactive chat for testing your model |
| `main.py` | Terminal chatbot (same as test.py with quit command) |
| `runclean.py` | Interactive CSV data cleaner |
| `torch-debug.py` | Environment diagnostic — checks GPU, CUDA, dependencies |
