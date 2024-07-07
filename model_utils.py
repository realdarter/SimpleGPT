import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
import logging
from encoderdecoder import *  # Import the tokenizer wrapper from encoder_decoder.py

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
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)

    # Add padding token if not already added
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"GPT-2 model (124M) downloaded and saved in {save_directory}")
    return True


if __name__ == "__main__":
    save_path = 'checkpoint/run1'
    csv_path = os.path.join(save_path, 'cleaned.csv')
    encoded_csv_path = os.path.join(save_path, 'csv_encoded.txt')
    batch_size = 4
    num_epochs = 1
    learning_rate = 5e-5
    download_gpt2_124M(save_path)

    ensure_file_exists(csv_path)
    encode_csv(csv_path, encoded_csv_path, header=True)

    # Step 1: Load pre-trained model and tokenizer
    model_name_or_path = 'gpt2'
    model, tokenizer = load_pretrained_model(model_name_or_path, model_name_or_path)
