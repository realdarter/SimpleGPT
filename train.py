from model_utils import *
import os

if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    csv_path = os.path.join(model_path, 'cleaned.csv')
    encoded_txt_path = os.path.join(model_path, 'csv_encoded.txt')


    if (ensure_file_exists(csv_path, create_if_missing=False)):
        encode_csv(csv_path, encoded_txt_path, header=True)
    else:
        exit()

    # Load pre-trained model and tokenizer before attempting to save them
    model_name_or_path = 'gpt2'


    # Uncomment this line to train the model
    train_on_dataset(model_path, encoded_txt_path, num_epochs=2, batch_size=3)
