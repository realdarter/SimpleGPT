from model_utils import *
import os

if __name__ == "__main__":
    model_path = 'checkpoint/delete'
    #model_path = 'checkpoint/reddit'
    csv_path = os.path.join(model_path, 'cleaned.csv')

    # Prepare the CSV data
    args = create_training_args(
        num_epochs=3,
        batch_size=8,
        learning_rate=3e-5,
        save_every=1000,
        max_length=512,
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2
    )

    # Train the model
    train_model(model_path, csv_path, args)

