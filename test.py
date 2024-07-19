from model_utils import *
import re
from tokenization import *

if __name__ == "__main__":
    model_directory = 'checkpoint/lgbtqsave'  # Replace with your actual model directory
    while True:
        prompt_text = input("Input: ")

        response = generate_responses(model_directory, prompt_text, args=create_training_args(), clean_result=True)

        print(f"Prompt: {prompt_text}")
        print(f"Generated Response: {response}")