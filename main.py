import logging
import pandas as pd
from data_utils import *
from download_utils import *
from model_utils import *

# Initialize logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        # Ensure GPT-2 model is downloaded
        if not (is_gpt2_downloaded()):
            download_gpt2()
        

        # Define file path and read data
        file_path = 'cleaned.csv'
        df = read_file(file_path)

        print("Preprocessing Data")
        # Preprocess data
        tokenized_data = preprocess_data(df)

        print("Training Data")
        # Train model
        trained_model = train_model(tokenized_data)

        print("Save Model")
        # Save model
        save_model(trained_model)

        print("Load Model")
        # Load model (example)
        loaded_model = load_model()

        # Example usage of loaded model (assuming tokenizer is imported in main.py)
        context = "Hello!"
        input_ids = tokenizer.encode(context, return_tensors='pt')
        generated_output = loaded_model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        print("Generated Reply:", generated_text)

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
