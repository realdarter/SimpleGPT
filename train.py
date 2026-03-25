from chat_gen import create_args, train_model
import discord_notify

if __name__ == "__main__":
    model_directory = 'checkpoint/run'
    csv_path = 'training_data.csv'

    print(f"[Output Processing Location] Discord Channel: {discord_notify.DISCORD_CHANNEL_ID}")
    success = discord_notify.send(f"[Output Processing Location] Discord Channel: {discord_notify.DISCORD_CHANNEL_ID}")
    print(f"[Discord] Message {'sent' if success else 'FAILED to send'}")

    args = create_args(
        # Training settings (batch_size, max_length, warmup auto-tuned from VRAM)
        save_every=3000,
        learning_rate=2e-4,
        patience=3,
        max_epochs=15,
        max_length=256,
        log_every=25,
        target_effective_batch_size=32,
        max_eval_samples=1536,
        write_diagnostics=True,
        sample_preview_count=3,
        sample_log_every_epochs=1,
        sample_max_new_tokens=96,
        # Generation settings (only matter when testing, not during training)
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2,
        enableSampleMode=False,
    )

    train_model(model_directory, csv_path, args)
