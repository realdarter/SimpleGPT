from chat_gen import create_args, train_model
import discord_notify


def on_event(event_type, **kwargs):
    """Forward training events to Discord."""
    if event_type == "training_start":
        discord_notify.notify_training_start(
            kwargs["model_dir"], kwargs["dataset_size"],
            kwargs["batch_size"], kwargs["max_epochs"], kwargs["auto_stop"]
        )
    elif event_type == "epoch_done":
        discord_notify.notify_epoch(
            kwargs["epoch"], kwargs["max_epochs"], kwargs["train_loss"],
            val_loss=kwargs.get("val_loss"), duration=kwargs.get("duration"),
            best=kwargs.get("best", False)
        )
    elif event_type == "training_done":
        discord_notify.notify_training_done(
            kwargs["total_time"], best_epoch=kwargs.get("best_epoch"),
            best_val_loss=kwargs.get("best_val_loss"),
            stopped_early=kwargs.get("stopped_early", False)
        )
    elif event_type == "sample":
        discord_notify.send(
            f"**Sample Generation** (Epoch {kwargs['epoch']}, Step {kwargs['step']})\n"
            f"**Prompt:** {kwargs['prompt']}\n"
            f"**Response:** {kwargs['response']}"
        )
    elif event_type == "sample_after_epoch":
        discord_notify.send(
            f"**Sample after epoch {kwargs['epoch']}**\n"
            f"> **Prompt:** {kwargs['prompt']}\n"
            f"> **Expected:** {kwargs['expected']}\n"
            f"> **Model:** {kwargs['generated']}"
        )
    elif event_type == "checkpoint":
        discord_notify.notify_checkpoint(kwargs.get("epoch", 0))


if __name__ == "__main__":
    model_directory = 'checkpoint/run'
    csv_path = 'training_data.csv'

    print(f"[Output Processing Location] Discord Channel: {discord_notify.DISCORD_CHANNEL_ID}")
    success = discord_notify.send(f"[Output Processing Location] Discord Channel: {discord_notify.DISCORD_CHANNEL_ID}")
    print(f"[Discord] Message {'sent' if success else 'FAILED to send'}")

    args = create_args(
        # Training settings (batch_size, max_length, warmup auto-tuned from VRAM)
        save_every=5000,
        learning_rate=5e-5,
        patience=2,
        max_epochs=5,
        max_length=256,
        log_every=25,
        target_effective_batch_size=32,
        max_eval_samples=1536,
        write_diagnostics=True,
        sample_preview_count=3,
        sample_log_every_epochs=1,
        sample_max_new_tokens=40,
        # Generation settings (only matter when testing, not during training)
        temperature=0.65,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.22,
        enableSampleMode=False,
    )

    train_model(model_directory, csv_path, args, on_event=on_event)
