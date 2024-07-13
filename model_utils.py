import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenization import *  
from file_utils import *
import time
from collections import deque

CUDA_LAUNCH_BLOCKING=1

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        print(f"Dataset size: {len(self.input_ids)}")  # Add this line
        print(f"Dataset tokens: {(self.input_ids[0])}")  # Add this line
        print(f"labels size: {len(self.labels)}")  # Add this line
        print(f"labels tokens: {(self.labels)[0]}")  # Add this line
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def load_model_and_tokenizer(model_directory):
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def download_gpt2_124M(save_directory):
    if check_gpt2_models_exist(save_directory):
        print("Model already exists. Not downloading.")
        return False
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"GPT-2 model (124M) downloaded and saved in {save_directory}")
    return True

def __print_training_progress__(epoch, num_epochs, i, len_dataloader, loss, avg_loss, start_time, total_steps):
    elapsed_time = time.time() - start_time
    steps_completed = epoch * len_dataloader + i
    steps_remaining = total_steps - steps_completed
    avg_time_per_step = elapsed_time / steps_completed
    estimated_time_remaining = avg_time_per_step * steps_remaining

    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len_dataloader}], Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f} seconds, Estimated Time Remaining: {estimated_time_remaining:.2f} seconds")

def train_model(model_directory, dataset_array=[], num_epochs=1, batch_size=1, save_every=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(model_directory)

    ensure_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, dataset_array)
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = len(dataloader) * num_epochs
    start_time = time.time()

    model.train()
    recent_losses = deque(maxlen=20)  # Keeps track of last 20 losses

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_start_time = time.time()

        for i, batch in enumerate(dataloader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            avg_loss = total_loss / i

            # Append current loss to recent_losses
            recent_losses.append(loss.item())

            __print_training_progress__(epoch, num_epochs, i, len(dataloader), loss.item(), avg_loss, start_time, total_steps)

            if i % save_every == 0:
                model.save_pretrained(model_directory)
                print(f"Model saved at iteration {i} in {model_directory}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_elapsed_time:.0f} seconds.")

        model.save_pretrained(model_directory)

    total_elapsed_time = time.time() - start_time
    print(f"Training completed in {total_elapsed_time // 60:.0f} minutes and {total_elapsed_time % 60:.0f} seconds.")













def test_input(model_directory, prompt_text, max_length=50, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.5):
    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_directory)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Add special tokens to the tokenizer if they are not already present
    special_tokens_dict = {'pad_token': '[PAD]', 'sep_token': '<[SEP]>', 'eos_token': '<[EOS]>', 'bos_token': '<[BOS]>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Encode the prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Generate text using the model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )


    # Decode the generated text
    #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text_special = tokenizer.decode(output[0], skip_special_tokens=False)

    #print(f"Generated Text: {generated_text}")
    #print(f"Generated Text Special: {generated_text_special}")
    return generated_text_special

def generate_responses(model_directory, prompt_text, max_length=50, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """
    Calls test_input to generate text and processes the output to separate the prompt and the responses.
    Returns a list where the first element is the prompt and the rest are the generated responses.
    """

    # Call the test_input function to generate text
    generated_text_special = test_input(
        model_directory, prompt_text, max_length, temperature, top_k, top_p, repetition_penalty
    )

    # Split the generated text into prompt and responses
    print(generated_text_special)
    before_tsep, sep, after_tsep = generated_text_special.partition('<|SEP|>')
    special_tokens_dict = {'pad_token': '<[PAD]>', 'sep_token': '<|SEP|>', 'eos_token': '<|EOS|>', 'bos_token': '<|BOS|>'}
    tokens_to_remove = special_tokens_dict.keys()

    for token in tokens_to_remove:
        before_tsep = before_tsep.replace(special_tokens_dict[token], '')
        after_tsep = after_tsep.replace(special_tokens_dict[token], '')

    return [before_tsep.strip(), after_tsep.strip()]

