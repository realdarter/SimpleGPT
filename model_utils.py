import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
import logging
from tokenization import *  # Import the tokenizer wrapper from encoder_decoder.py
import os
from file_utils import *

model = None
tokenizer = None

def load_pretrained_model(model_name_or_path, tokenizer_name_or_path):
    """
    Loads a pre-trained GPT-2 model and tokenizer.
    Args:
    - model_name_or_path (str): Name or path of the pre-trained model.
    - tokenizer_name_or_path (str): Name or path of the tokenizer associated with the model.
    Returns:
    - GPT2LMHeadModel: The loaded GPT-2 model.
    - GPT2Tokenizer: The loaded GPT-2 tokenizer.
    """
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


def train_on_dataset(model_directory, dataset_path, num_epochs=1, max_length=512, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    global model, tokenizer

    context_texts = read_txt_file(dataset_path)
    tokenized_dataset = tokenize_dataset(tokenizer, context_texts, max_length=max_length)
    dataloader = create_dataloader(tokenized_dataset, batch_size=batch_size)
    print(tokenized_dataset[0])
    #print(decode_tokenize_text(tokenizer, tokenized_dataset[0], skip_special_tokens=False, remove_pad=True))

    model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    print(f"Training complete.")

if __name__ == "__main__":
    save_path = 'checkpoint/run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_txt_path = os.path.join(save_path, 'csv_encoded.txt')
    batch_size = 4
    num_epochs = 1
    learning_rate = 5e-5

    # Load pre-trained model and tokenizer before attempting to save them
    model_name_or_path = 'gpt2'
    model, tokenizer = load_pretrained_model(model_name_or_path, model_name_or_path)

    download_gpt2_124M(save_path)

    ensure_file_exists(csv_path)
    encode_csv(csv_path, encoded_txt_path, header=True)

    train_on_dataset(save_path, encoded_txt_path, num_epochs=num_epochs, batch_size=batch_size)