import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_reply(context, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(context, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('gpt2_model.pth', map_location=device))
    model.to(device)

    context = "Hello, how are you?"
    reply = generate_reply(context, model, tokenizer, device)
    print(f"Reply: {reply}")
