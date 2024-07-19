from model_utils import *
from file_utils import *
import os
CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":
    model_path = 'checkpoint/lgbtqsave2'
    #model_path = 'checkpoint/reddit'
    csv_path = os.path.join(model_path, 'cleaned.csv')

    # Prepare the CSV data
    encoded_data = prepare_csv(csv_path)

    # Train the model
    train_model(model_path, encoded_data, num_epochs=3, batch_size=4, save_every=500)

