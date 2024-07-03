import os
import time
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

# Special tokens


class TextDataset(Dataset):
    """Custom dataset class for text data."""
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

def initialize_gpt2_model(model_name='gpt2'):
    """Initialize a GPT-2 model and tokenizer from scratch."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Resize model embeddings to accommodate the new pad token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train_on_dataset(model, tokenizer, dataset_path, num_epochs=1, batch_size=8, save_every=100, save_directory="checkpoint"):
    """Train the model on the provided dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    df = pd.read_csv(dataset_path)

    dataset = TextDataset(df['context'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_batches = len(dataloader)
    print(f"Total batches per epoch: {total_batches}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch.to(device)
            labels = inputs.clone()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

            if (batch_idx + 1) % save_every == 0 or (batch_idx + 1) == total_batches:
                checkpoint_dir = os.path.join(save_directory, "checkpoint")
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

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

def test_input(model, tokenizer, prompt_text, temperature=0.7):
    """Generate text based on a prompt using the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    input_ids = tokenizer.encode(f" {prompt_text} ", return_tensors='pt').to(device)

    max_length = 100
    eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = eos_token_id
    attention_mask = torch.ones(input_ids.shape, device=device)

    beam_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        temperature=temperature,
        do_sample=True
    )

    generated_texts = []
    for beam_idx, beam in enumerate(beam_output):
        # Decode output without skipping special tokens initially
        decoded_beam = tokenizer.decode(beam, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        # Split decoded beam into tokens
        tokens = decoded_beam.split()
        
        # Check if '<|endoftext|>' token is in tokens
        has_endoftext = ('<|endoftext|>' in tokens)

        # Print tokens for inspection
        print(f"Tokens for Beam {beam_idx + 1}:")
        print(tokens)

        generated_texts.append(decoded_beam.replace('[NextLine]', '\n'))

        print(f"Beam {beam_idx + 1}:")
        print(f" {prompt_text}  {generated_texts[-1]}")
        print(f"Has endoftext token: {has_endoftext}")

    return generated_texts


if __name__ == "__main__":
    print("Running Code")
    dataset_path = "cleaned.csv"
    save_directory = "checkpoint/run2"

    model, tokenizer = initialize_gpt2_model('gpt2')
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    print(f"End of text token: {eos_token}")
    print(f"End of text token ID: {eos_token_id}")
    train_on_dataset(model, tokenizer, dataset_path, num_epochs=30, batch_size=4, save_every=100, save_directory=save_directory)

    while True:
        inp = input("Ask Question: ")
        test_input(model, tokenizer, inp)
