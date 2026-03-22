from chat_gen import create_args, train_model

if __name__ == "__main__":
    model_directory = 'checkpoint/run3'
    csv_path = 'training_data.csv'

    args = create_args(
        # Training settings (batch_size, max_length, save_every, warmup auto-tuned from VRAM)
        learning_rate=2e-4,
        patience=3,
        max_epochs=15,
        # Generation settings (only matter when testing, not during training)
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2,
        enableSampleMode=True,
    )

    train_model(model_directory, csv_path, args)
