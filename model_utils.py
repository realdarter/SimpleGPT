import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
import logging
from exceptions import ModelTrainingError, ModelNotSaved, ModelNotLoaded

logging.basicConfig(level=logging.INFO)

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def train_model(tokenized_data, num_epochs=3, learning_rate=5e-5):
    try:
        dataset = GPT2Dataset(tokenized_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        for epoch in range(num_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].squeeze().to(device)
                attention_mask = batch['attention_mask'].squeeze().to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        return model
    except Exception as e:
        raise ModelTrainingError(f"Error training GPT-2 model: {str(e)}")

def save_model(model, model_name='gpt2', path='models/124M'):
    try:
        model.save_pretrained(path)
        logging.info(f"Model '{model_name}' saved successfully.")
    except Exception as e:
        raise ModelNotSaved(model_name)

def load_model(model_name='gpt2', path='models/124M'):
    try:
        model = GPT2LMHeadModel.from_pretrained(path)
        logging.info(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        raise ModelNotLoaded(model_name)
