import torch
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd

class GPT2Dataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def train_model(tokenized_data, num_epochs=3, learning_rate=5e-5):
    dataset = GPT2Dataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch[0]['input_ids'].to(device)
            labels = batch[0]['input_ids'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

if __name__ == "__main__":
    file_path = 'your_data.csv'
    from data_preprocessing import load_and_preprocess_data
    tokenized_data = load_and_preprocess_data(file_path)
    model = train_model(tokenized_data)
    # Save the model if needed
    torch.save(model.state_dict(), 'gpt2_model.pth')
