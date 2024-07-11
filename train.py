from model_utils import *
import os
CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    csv_path = os.path.join(model_path, 'cleaned.csv')
    csv_path = 'cleaned.csv'
    #encoded_txt_path = os.path.join(model_path, 'csv_encoded.txt')


    encoded_data = prepare_csv(csv_path)
    print(encoded_data[0])
    print(encoded_data[-1])

    train_model(model_path, encoded_data, num_epochs=10, batch_size=3, save_every=100)

