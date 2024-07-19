import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenization import *  
from file_utils import *
import time
from collections import deque

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

# Function to create training arguments
def create_training_args(num_epochs=1, batch_size=1, learning_rate=5e-5, save_every=500, 
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


def load_model_and_tokenizer(model_directory):
    """
    Load the pre-trained GPT-2 model and tokenizer from a specified directory.
    Args: model_directory (str): Directory containing the pre-trained model and tokenizer.
    Returns: tuple: A tuple containing the model and tokenizer.
    """
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    model.resize_token_embeddings(len(tokenizer))

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

def train_model(model_directory=None, csv_path=None, args=create_training_args()):
    """
    Train the GPT-2 model on a custom dataset of context-response pairs.

    Args:
        model_directory (str, optional): Directory where the model is saved.
        csv_path (str, optional): Path to the CSV file containing context-response pairs.
        args (dict, optional): Dictionary of training arguments.
    """

    if model_directory is None:
        print("No given Model Directory")
        return
    if csv_path is None:
        print("No given CSV Path")
        return
    if args is None:
        print("Error with Training Args")
        return
    
    num_epochs = args.get('num_epochs')
    batch_size = args.get('batch_size')
    save_every = args.get('save_every')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    download_gpt2_124M(model_directory)

    model, tokenizer = load_model_and_tokenizer(model_directory)

    ensure_tokens(model, tokenizer)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    encoded_data = prepare_csv(csv_path,start_token=tokenizer.eos_token, sep_token=tokenizer.sep_token)

    input_ids, attention_masks, labels = tokenize_dataset(tokenizer, encoded_data)
    dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    print(pad_token)
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

def generate_responses(model_directory, prompt_text, args=create_training_args(), clean_result=False):
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
    model, tokenizer = load_model_and_tokenizer(model_directory)
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
