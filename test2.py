import os
import time
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_text = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        
        # Pad sequences to ensure each batch has the same length
        tokenized_text += [self.tokenizer.pad_token_id] * (self.max_length - len(tokenized_text))

        return torch.tensor(tokenized_text, dtype=torch.long)

# Function to initialize GPT-2 model from scratch
def initialize_gpt2_model():
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Resize model embeddings to accommodate the new pad token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

# Function to tokenize and train on cleaned.csv
def train_on_dataset(model, tokenizer, dataset_path, num_epochs=1, batch_size=8, save_every=500):
    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Load the dataset using Pandas
    df = pd.read_csv(dataset_path)

    # Create a Dataset and DataLoader
    dataset = TextDataset(df['context'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_batches = len(dataloader)
    print(f"Total batches per epoch: {total_batches}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            inputs = batch.to(device)

            # Labels are shifted inputs
            labels = inputs.clone()

            # Model forward pass
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print current progress every 50 batches
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                elapsed_time = time.time() - start_time
                batches_done = batch_idx + 1
                batches_left = total_batches - batches_done
                avg_time_per_batch = elapsed_time / batches_done
                estimated_time_remaining = batches_left * avg_time_per_batch

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batches_done}/{total_batches}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Elapsed Time: {elapsed_time:.2f} sec, "
                      f"ETA: {estimated_time_remaining:.2f} sec remaining")

            # Save model checkpoint every 'save_every' batches
            if (batch_idx + 1) % save_every == 0 or (batch_idx + 1) == total_batches:
                checkpoint_dir = os.path.join(model_directory, "checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint at epoch {epoch+1}, batch {batches_done}")

        end_time = time.time()
        epoch_duration = end_time - start_time

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        print(f"Time taken for epoch {epoch+1}: {epoch_duration:.2f} seconds")

    # Save final model after training
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

def test_input(model, tokenizer, prompt_text, temperature=0.7):
    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Tokenize the input text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    # Generate text using the model
    max_length = 100
    eos_token_id = tokenizer.eos_token_id

    # Ensure the eos_token_id is in the model's config
    model.config.pad_token_id = eos_token_id

    # Generate attention mask
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Generate text
    beam_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,  # Increase num_beams for better diversity
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        temperature=temperature,
        do_sample=True  # Ensure do_sample is set to True
    )

    # Decode the generated output
    decoded_beam = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Print the final generated output
    print("Beam Output:")
    print(decoded_beam.replace('[NextLine]', '\n'))  # Replace [NextLine] tokens with newlines

# Main function
if __name__ == "__main__":
    print("Running Code")
    save_directory = "checkpoint/run1"
    dataset_path = "cleaned.csv"

    # Step 1: Initialize GPT-2 model and tokenizer from scratch
    model, tokenizer = initialize_gpt2_model()

    # Step 2: Train on dataset
    train_on_dataset(model, tokenizer, dataset_path, num_epochs=10, batch_size=4, save_every=500)

    # Step 3: Test input
    while True:
        inp = input("Ask Question: ")
        test_input(model, tokenizer, inp)
