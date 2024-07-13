from model_utils import *
import re
from tokenization import *


if __name__ == "__main__":
    model_directory = 'checkpoint/run1'  # Replace with your actual model directory
    while True:
        prompt_text = input("Input: ")
        prompt_text = f"<[BOS]> {prompt_text} <[SEP]>"
        
        responses = generate_responses(model_directory, prompt_text)
        prompt = responses[0]
        generated_response = responses[1]

        print(f"Prompt: {prompt}")
        print(f"Generated Response: {generated_response}")