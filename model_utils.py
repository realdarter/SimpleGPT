import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tokenization import *  # Import the tokenizer wrapper from encoder_decoder.py
from file_utils import *
import time
import random

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def check_gpt2_models_exist(model_path):
    model_files = [
        'added_tokens.json',
        'config.json',
        'generation_config.json',
        'merges.txt',
        'model.safetensors',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.json'
    ]
    all_files_exist = True
    for file in model_files:
        file_path = os.path.join(model_path, file)
        absolute_path = os.path.abspath(file_path)  # Get the absolute path
        if not os.path.isfile(absolute_path):
            print(f"File missing: {absolute_path}")
            all_files_exist = False
    return all_files_exist

def download_gpt2_124M(save_directory):
    if check_gpt2_models_exist(save_directory):
        print("Model already exists. Not downloading.")
        return False
    global model, tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"GPT-2 model (124M) downloaded and saved in {save_directory}")
    return True

def train_model(model_directory, dataset_path, num_epochs=1, batch_size=1, save_every=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    print("Loading Data...")
    textlines = read_txt_file(dataset_path)  # Assuming this returns an array of texts
    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, textlines)
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Finished Loading Data")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    model.train()

    start_time = time.time()  # Start timing the training process
    total_steps = len(dataloader) * num_epochs

    iterations = 0
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            steps_completed = epoch * len(dataloader) + i + 1
            steps_remaining = total_steps - steps_completed
            avg_time_per_step = elapsed_time / steps_completed
            estimated_time_remaining = avg_time_per_step * steps_remaining

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f} seconds, Estimated Time Remaining: {estimated_time_remaining:.2f} seconds")

            iterations += 1
            if iterations % save_every == 0:
                model.save_pretrained(model_directory)
                print(f"Model saved at iteration {iterations} in {model_directory}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")

    end_time = time.time()  # End timing after training completes
    elapsed_time = end_time - start_time

    print(f"Training completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds.")

    model.save_pretrained(model_directory)
    print(f"Final model saved in {model_directory}")

def test_input(model_directory, prompt_text, max_length=50, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.5):
    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Add special tokens to the tokenizer if they are not already present
    special_tokens_dict = {'pad_token': '[PAD]', 'sep_token': '<|septext|>', 'eos_token': '<|endoftext|>', 'bos_token': '<|startoftext|>'}
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
    before_tsep, sep, after_tsep = generated_text_special.partition('<|septext|>')
    special_tokens_dict = {'pad_token': '[PAD]', 'sep_token': '<|septext|>', 'eos_token': '<|endoftext|>', 'bos_token': '<|startoftext|>'}
    tokens_to_remove = special_tokens_dict.keys()

    for token in tokens_to_remove:
        before_tsep = before_tsep.replace(special_tokens_dict[token], '')
        after_tsep = after_tsep.replace(special_tokens_dict[token], '')

    return [before_tsep.strip(), after_tsep.strip()]


"""
if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    csv_path = os.path.join(model_path, 'cleaned.csv')
    encoded_txt_path = os.path.join(model_path, 'csv_encoded.txt')
    batch_size = 1  # Reduce batch size to fit within memory
    if (ensure_file_exists(csv_path, create_if_missing=False)):
        encode_csv(csv_path, encoded_txt_path, header=True)
    else:
        exit()

    # Load pre-trained model and tokenizer before attempting to save them
    model_name_or_path = 'gpt2'
    #check_gpt2_models_exist(save_path)
    download_gpt2_124M(save_path)


    # Uncomment this line to train the model
    #train_on_dataset(save_path, encoded_txt_path, num_epochs=1, batch_size=3)

    while True:
        test_prompt = input("Input: ")
        test_prompt = f"<|startoftext|> {test_prompt} <|septext|>"
        test_input(model_path, test_prompt)
"""