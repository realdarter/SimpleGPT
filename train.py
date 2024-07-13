from model_utils import *
from file_utils import *
import os
CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    csv_path = os.path.join(model_path, 'cleaned.csv')

    # Prepare the CSV data
    encoded_data = prepare_csv(csv_path)

    download_gpt2_124M(model_path)
    # Train the model
    train_model(model_path, encoded_data, num_epochs=50, batch_size=3, save_every=100)

