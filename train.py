import os
from chat_gen import create_args, train_model

if __name__ == "__main__":
    model_directory = 'checkpoint/run3'
    csv_path = os.path.join(model_directory, 'cleaned.csv')

    args = create_args(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        save_every=500,
        max_length=512,
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2,
        enableSampleMode=True,
        warmup_steps=100
    )

    train_model(model_directory, csv_path, args)
