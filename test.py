from model_utils import *
import re
from tokenization import *

if __name__ == "__main__":
    model_directory = 'checkpoint/lgbtqsave'  # Replace with your actual model directory
    while True:
        prompt_text = input("Input: ")
        prompt_text = f"<[BOS]> {prompt_text} <[SEP]>"
        #prompt_text = f"{prompt_text} <[SEP]>"
        responses = generate_responses(model_directory, prompt_text, repetition_penalty=2.0)

        model, tokenizer = load_model_and_tokenizer(model_directory)
        ensure_tokens(tokenizer)
        print(tokenizer.bos_token_id)
        bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
        print(f"BOS Token Text: {bos_token}")

        prompt = responses[0]
        generated_response = responses[1]
        print(f"Prompt: {prompt}")
        print(f"Generated Response: {generated_response}")