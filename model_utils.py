import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tokenization import *  # Import the tokenizer wrapper from encoder_decoder.py
from file_utils import *
import time

model = None
tokenizer = None

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
def load_pretrained_model(model_name_or_path, tokenizer_name_or_path):
    global model, tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    ensure_pad_token(tokenizer)
    return model, tokenizer

def check_gpt2_models_exist(model_path):
    model_files = [
        'config.json',
        'merges.txt',
        'pytorch_model.bin',
        'tokenizer.json',
        'training_args.bin',
        'vocab.json',
        'vocab.txt'
    ]
    models_exist = all(os.path.isfile(os.path.join(model_path, f'gpt2-{size}', file)) for size in ['small', 'medium', 'large', 'xl'] for file in model_files)
    return models_exist

def download_gpt2_124M(save_directory):
    if check_gpt2_models_exist(save_directory):
        return False
    global model, tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"GPT-2 model (124M) downloaded and saved in {save_directory}")
    return True

def train_on_dataset(model_directory, dataset_path, num_epochs=1, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_pretrained_model(model_directory, model_directory)
    ensure_pad_token(tokenizer)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    textlines = read_txt_file(dataset_path)  # Assuming this returns an array of texts
    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, textlines)
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    model.train()

    start_time = time.time()  # Start timing the training process

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
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}, Elapsed Time: {elapsed_time:.2f} seconds")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")

    end_time = time.time()  # End timing after training completes
    elapsed_time = end_time - start_time

    print(f"Training completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds.")

    model.save_pretrained(model_directory)

def test_input(model_directory, prompt_text):
    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Tokenize the input text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    # Generate text based on the input using beam search
    num_beams = 2
    beam_output = model.generate(input_ids,
                                 max_length=100,
                                 num_beams=num_beams,
                                 num_return_sequences=num_beams,
                                 no_repeat_ngram_size=2,
                                 early_stopping=True,
                                 pad_token_id=tokenizer.eos_token_id)

    # Print and decode the beam search outputs
    print("Beam Output:")
    skipped_tokens = []  # List to store tokens that couldn't be decoded
    eos_token_id = tokenizer.eos_token_id
    print(f"eos_token_id = {eos_token_id}")

    for i, beam in enumerate(beam_output):
        # Initialize list to store decoded tokens
        decoded_tokens = []
        
        # Filter out None and padding tokens
        filtered_beam = [token for token in beam.tolist() if token is not None and token != tokenizer.pad_token_id and token not in skipped_tokens]

        for token in filtered_beam:
            if token == eos_token_id:
                break  # Stop decoding once EOS token is encountered

            try:
                decoded_token = tokenizer.decode(token, skip_special_tokens=False)
                decoded_tokens.append(decoded_token)
            except Exception as e:
                print(f"Error decoding token {token}: {e}")
                skipped_tokens.append(token)  # Add token to skipped list

        # Join decoded tokens into a single string
        decoded_beam = " ".join(decoded_tokens)
        
        print(f"Beam {i}: {decoded_beam}")
        print()

if __name__ == "__main__":
    save_path = 'checkpoint/run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_txt_path = os.path.join(save_path, 'csv_encoded.txt')
    batch_size = 1  # Reduce batch size to fit within memory
    num_epochs = 10

    # Load pre-trained model and tokenizer before attempting to save them
    model_name_or_path = 'gpt2'
    model, tokenizer = load_pretrained_model(model_name_or_path, model_name_or_path)

    download_gpt2_124M(save_path)
    ensure_file_exists(csv_path)
    encode_csv(csv_path, encoded_txt_path, header=True)

    train_on_dataset(save_path, encoded_txt_path, num_epochs=num_epochs, batch_size=batch_size)

    test_prompt = "What are your thoughts on artificial intelligence?"
    test_input(save_path, test_prompt)