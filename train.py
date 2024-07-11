from model_utils import *
import os

if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    csv_path = os.path.join(model_path, 'cleaned.csv')
    encoded_txt_path = os.path.join(model_path, 'csv_encoded.txt')


    if (ensure_file_exists(csv_path, create_if_missing=False)):
        prepare_csv(csv_path, encoded_txt_path, header=True)
    else:
        exit()

    model_name_or_path = 'gpt2'

    train_model(model_path, encoded_txt_path, num_epochs=10, batch_size=3)

