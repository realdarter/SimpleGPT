"""
Coded By Goose 🪿
"""
import torch
from torch import tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
from collections import deque
import os
import pandas as pd


CUDA_LAUNCH_BLOCKING=1

# Custom Dataset class for PyTorch DataLoader
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


def ensure_file_exists(file_path, create_if_missing=True):
    """
    Ensures the directory for the specified file path exists. 
    If the file does not exist, creates the necessary directories and an empty file if create_if_missing is True.
    Args:
    - file_path (str): The path of the file to check or create.
    - create_if_missing (bool): Whether to create the file if it does not exist. Defaults to True.
    Returns: bool: True if the file already exists or was successfully created, False if there was an error.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
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
    else:
        return False

def prepare_csv(csv_path, header=True, start_token='', sep_token=''):
    """ 
    Reads a CSV file and returns a list of all items with optional start, separator, and end tokens.
    
    Args:
        csv_path (str): The path to the CSV file to read.
        header (bool, optional): Whether the CSV file includes a header row. Defaults to True.
        start_token (str, optional): A token to prepend to each row's content. Defaults to ''.
        sep_token (str, optional): A token to insert between items in a row. Defaults to ''.
    
    Returns: 
        list: A list of formatted strings, each representing a row from the CSV file.
    """
    start_time = time.time()
    if header:
        df = pd.read_csv(csv_path, dtype=str)
    else:
        df = pd.read_csv(csv_path, header=None, dtype=str)

    # Drop NaNs and replace directly
    formatted_rows = df.fillna('').apply(lambda row: f"{start_token} " + f" {sep_token} ".join(row.str.strip().str.replace('"', '')), axis=1)
    all_items = formatted_rows.tolist()

    elapsed_time = time.time() - start_time
    print(f"Time taken to prepare CSV: {elapsed_time:.4f} seconds")
    print(formatted_rows[0])
    return all_items


def check_gpt2_models_exist(model_path):
    """
    Checks if all necessary files for a GPT-2 model exist in the specified directory.
    Args: model_path (str): The path to the directory containing model files.
    
    Returns: bool: True if all required files exist, False if any file is missing.
    """
    model_files = [
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

def tokenize_single_text(tokenizer, text, max_length=512):
    """
    Tokenizes a single line of text and ensures it is padded or truncated to a maximum length.
    Returns a dictionary with 'input_ids' and 'attention_mask'.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        text (str): The text to tokenize.
        max_length (int, optional): The maximum length of the tokenized sequence. Defaults to 512.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask'.
    """
    encoded_text = tokenizer.encode_plus(
        text.strip(), 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length', 
        return_tensors='pt'
    )
    return {
        'input_ids': encoded_text['input_ids'],
        'attention_mask': encoded_text['attention_mask']
    }

def tokenize_dataset(tokenizer, texts, max_length=512, eos_token="<[EOS]>"):
    """
    Tokenizes a list of texts and ensures each text is padded or truncated to a maximum length.
    Returns input_ids, attention_masks, and labels.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        texts (list of str or str): The texts to tokenize.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 512.
        eos_token (str, optional): The end-of-sequence token. Defaults to "<[EOS]>".

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Tensor of token IDs.
            - attention_masks (torch.Tensor): Tensor of attention masks.
            - labels (torch.Tensor): Tensor of labels for language modeling.
    """
    if isinstance(texts, str):
        texts = [texts]
        
    tokenized_texts = [tokenizer.encode_plus(text, max_length=max_length, truncation=True, padding='max_length') for text in texts]
    
    input_ids = torch.tensor([item['input_ids'] for item in tokenized_texts], dtype=torch.long)
    attention_masks = torch.tensor([item['attention_mask'] for item in tokenized_texts], dtype=torch.long)
    
    eos_ids = tokenizer.convert_tokens_to_ids(eos_token)
    eos_tensor = torch.full((input_ids.size(0), 1), eos_ids, dtype=torch.long)
    input_ids = torch.cat([input_ids, eos_tensor], dim=1)
    
    attention_masks = torch.cat([attention_masks, torch.ones((attention_masks.size(0), 1), dtype=torch.long)], dim=1)
    
    labels = input_ids.clone()
    
    return input_ids, attention_masks, labels



def ensure_tokens(model, tokenizer, pad_token='<[PAD]>', sep_token='<[SEP]>', eos_token='<[EOS]>', bos_token='<[BOS]>'):
    """
    Adds special tokens to the tokenizer and adjusts the model's token embeddings to account for these tokens.
    Args:
        model: The model object (e.g., a Hugging Face Transformers model) that needs its token embeddings resized.
        tokenizer: The tokenizer object (e.g., a Hugging Face Transformers tokenizer) to which special tokens will be added.
        pad_token (str, optional): The token used for padding sequences. Defaults to '<[PAD]>'.
        sep_token (str, optional): The token used to separate sequences. Defaults to '<[SEP]>'.
        eos_token (str, optional): The token used to indicate the end of a sequence. Defaults to '<[EOS]>'.
        bos_token (str, optional): The token used to indicate the beginning of a sequence. Defaults to '<[BOS]>'.
    Returns: None
    """
    special_tokens_dict = {
        'pad_token': pad_token,
        'sep_token': sep_token,
        'eos_token': eos_token,
        'bos_token': bos_token
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model.resize_token_embeddings(len(tokenizer))

def decode_data(tokenizer, token_ids, skip_special_tokens=True):
    """
    Decodes a list or tensor of token IDs back into a string.
    
    Args:
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        token_ids (list of int or torch.Tensor): The token IDs to decode.
        skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

    Returns:
        str: The decoded string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        raise ValueError("token_ids should be a list or a tensor of integers.")
    
    decoded_data = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return decoded_data

# Function to create training arguments
def create_args(num_epochs=1, batch_size=1, learning_rate=5e-5, save_every=500, 
                           max_length=512, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """
    Returns a dictionary of training arguments.
    Args:
        num_epochs (int, optional): Number of epochs. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to 1.
        learning_rate (float, optional): Learning rate. Defaults to 5e-5.
        save_every (int, optional): Save model every X steps. Defaults to 500.
        max_length (int, optional): Maximum length of generated sequences. Defaults to 512.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_k (int, optional): Top-K sampling. Defaults to 50.
        top_p (float, optional): Top-P (nucleus) sampling. Defaults to 0.95.
        repetition_penalty (float, optional): Repetition penalty. Defaults to 1.2.

    Returns:
        dict: Dictionary containing training arguments.
    """
    return {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }



def load_model_and_tokenizer(model_directory, download=True):
    start_time = time.time()
    if (not check_gpt2_models_exist(model_directory) and download):
        download_gpt2_124M(model_directory)
    print(model_directory)
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    model.resize_token_embeddings(len(tokenizer))
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")
    return model, tokenizer

def download_gpt2_124M(save_directory):
    """
    Download and save the GPT-2 124M model and tokenizer if they do not already exist.
    Args: save_directory (str): Directory where the model and tokenizer should be saved.
    Returns: bool: True if the model was downloaded, False if it already existed.
    """
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
    """
    Print the training progress including elapsed time, loss, and estimated time remaining.
    Args: 
        epoch (int): Current epoch number. 
        num_epochs (int): Total number of epochs.
        i (int): Current step number within the epoch.
        len_dataloader (int): Total number of steps in the dataloader.
        loss (float): Current loss value.
        avg_loss (float): Average loss up to the current step.
        start_time (float): Start time of the training process.
        total_steps (int): Total number of steps in training.
    """

    elapsed_time = time.time() - start_time
    steps_completed = epoch * len_dataloader + i
    steps_remaining = total_steps - steps_completed
    avg_time_per_step = elapsed_time / steps_completed
    estimated_time_remaining = avg_time_per_step * steps_remaining

    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len_dataloader}], Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f} seconds, Estimated Time Remaining: {estimated_time_remaining:.2f} seconds")

def train_model(model_directory=None, csv_directory=None, args=create_args()):
    """
    Train the GPT-2 model on a custom dataset of context-response pairs.

    Args:
        model_directory (str, optional): Directory where the model is saved.
        csv_path (str, optional): Path to the CSV file containing context-response pairs.
        args (dict, optional): Dictionary of training arguments.
    """


    if csv_directory is None:
        print("No given CSV Path")
        return
    if model_directory is None:
        print("No Model Path")
        return
    if args is None:
        print("Error with Training Args")
        return
    
    model, tokenizer = load_model_and_tokenizer(model_directory)

    num_epochs = args.get('num_epochs')
    batch_size = args.get('batch_size')
    save_every = args.get('save_every')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    #model, tokenizer = load_model_and_tokenizer(model_directory)

    ensure_tokens(model, tokenizer)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    encoded_data = prepare_csv(csv_directory,start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)

    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, encoded_data)
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    first_decoded = decode_data(tokenizer, input_ids[0], skip_special_tokens=False)
    print("First training example (with special tokens):")
    print(first_decoded)

    total_steps = len(dataloader) * num_epochs
    start_time = time.time()
    print("Training")
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

def clean_text(uncleaned_text,  pad_token = '', sep_token = '', eos_token = '', bos_token = ''):
    """
    Clean and format the generated text by removing unwanted tokens.
    Args:
        uncleaned_text (str): The raw text generated by the model.
        pad_token (str, optional): Padding token to be removed. Defaults to ''.
        sep_token (str, optional): Separator token to be removed. Defaults to ''.
        eos_token (str, optional): End-of-sequence token to be removed. Defaults to ''.
        bos_token (str, optional): Beginning-of-sequence token to be removed. Defaults to ''.
    Returns: str: Cleaned text with unwanted tokens removed.
    """

    special_tokens_dict = {'pad_token': pad_token, 'sep_token': sep_token, 'eos_token': eos_token, 'bos_token': bos_token}
    tokens_to_remove = special_tokens_dict.values()
    before_tsep, sep, after_tsep = uncleaned_text.partition(sep_token)
    
    after_tsep = after_tsep.replace(bos_token, '').strip()
    while after_tsep.startswith(sep_token) or after_tsep.startswith(bos_token):
        if after_tsep.startswith(sep_token):
            after_tsep = after_tsep[len(sep_token):].strip()
        if after_tsep.startswith(bos_token):
            after_tsep = after_tsep[len(bos_token):].strip()
    split_text = after_tsep.split(sep_token)
    if len(split_text) == 1 and split_text[0] == split_text:
        split_text = [split_text.strip()]
    split_text = split_text[0]

    for token in tokens_to_remove:
        before_tsep = before_tsep.replace(token, '').strip()
        split_text = split_text.replace(token, '').strip()

    return split_text

def format_prompt(prompt_text, start_token="", sep_token=""):
    return f"{start_token} {prompt_text} {sep_token}"

def generate_responses(model, tokenizer, prompt_text, args=create_args(), clean_result=False):
    """
    Generate a response from the model given a prompt.
    Args:
        prompt (str): Input prompt for the model.
        model_directory (str): Directory containing the trained model and tokenizer.
        tokenizer (Tokenizer): Tokenizer for the model.
        max_length (int, optional): Maximum length of the generated text. Defaults to 512.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_k (int, optional): Top-K sampling. Defaults to 50.
        top_p (float, optional): Top-P sampling. Defaults to 0.95.
        repetition_penalty (float, optional): Repetition penalty. Defaults to 1.2.
    Returns: str: Generated response from the model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    ensure_tokens(model, tokenizer)
    model.to(device)

    prompt_text = format_prompt(prompt_text, start_token=tokenizer.bos_token, sep_token=tokenizer.sep_token)

    max_length = args.get('max_length')
    temperature = args.get('temperature')
    top_k = args.get('top_k')
    top_p = args.get('top_p')
    repetition_penalty = args.get('repetition_penalty')

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,  
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    generated_text_special = tokenizer.decode(output[0], skip_special_tokens=False)
    if clean_result:
        return clean_text(generated_text_special, pad_token = tokenizer.pad_token, sep_token = tokenizer.sep_token, eos_token = tokenizer.eos_token, bos_token = tokenizer.bos_token)
    return generated_text_special

