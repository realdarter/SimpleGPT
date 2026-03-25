"""
Discord notification helper for training progress.
Sends messages to a Discord channel via the Dara bot.
Based on CustomDiscordMessaging by dardar.
"""

import os

try:
    import requests
except ImportError:
    requests = None

DISCORD_CHANNEL_ID = "1346717893911511051"
_TOKEN_CACHE = None


def _read_token():
    global _TOKEN_CACHE
    if _TOKEN_CACHE is not None:
        return _TOKEN_CACHE
    auth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth.txt")
    if not os.path.isfile(auth_path):
        return None
    with open(auth_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token.startswith("Bot "):
        token = f"Bot {token}"
    _TOKEN_CACHE = token
    return _TOKEN_CACHE


def _get_headers(token):
    return {
        "Authorization": token,
        "Content-Type": "application/json",
    }


def send(message: str) -> bool:
    """Send a message to the training notifications channel. Returns True on success."""
    if requests is None:
        print("[Discord] requests is not installed; notifications are disabled.")
        return False

    token = _read_token()
    if token is None:
        return False

    payload = {
        "content": message,
        "tts": False,
    }
    try:
        response = requests.post(
            f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
            headers=_get_headers(token),
            json=payload
        )
        if response.status_code not in (200, 201):
            print(f"[Discord] Error {response.status_code}: {response.text[:200]}")
        return response.status_code in (200, 201)
    except requests.exceptions.RequestException as exc:
        print(f"[Discord] Request failed: {exc}")
        return False


def notify_checkpoint(epoch: int, train_loss: float = None, val_loss: float = None,
                      elapsed: str = None, sample: str = None):
    """Send a checkpoint save notification."""
    parts = [f"**Checkpoint saved** - Epoch {epoch}"]
    if train_loss is not None:
        parts.append(f"Train Loss: `{train_loss:.4f}`")
    if val_loss is not None:
        parts.append(f"Val Loss: `{val_loss:.4f}`")
    if elapsed:
        parts.append(f"Elapsed: {elapsed}")
    if sample:
        parts.append(f"Sample: {sample[:200]}")
    send("\n".join(parts))


def notify_epoch(epoch: int, max_epochs: int, train_loss: float, val_loss: float = None,
                 duration: str = None, best: bool = False):
    """Send an epoch completion notification."""
    star = " *New best!*" if best else ""
    parts = [f"**Epoch {epoch}/{max_epochs} complete**{star}"]
    parts.append(f"Train Loss: `{train_loss:.4f}`")
    if val_loss is not None:
        parts.append(f"Val Loss: `{val_loss:.4f}`")
    if duration:
        parts.append(f"Duration: {duration}")
    send("\n".join(parts))


def notify_training_start(model_dir: str, dataset_size: int, batch_size: int,
                          max_epochs: int, auto_stop: bool):
    """Send a training started notification."""
    mode = f"AUTO-STOP (max {max_epochs})" if auto_stop else f"FIXED ({max_epochs} epochs)"
    parts = [
        "**Training started**",
        f"Model: `{model_dir}`",
        f"Dataset: {dataset_size} samples, Batch: {batch_size}",
        f"Mode: {mode}",
    ]
    send("\n".join(parts))


def notify_training_done(total_time: str, best_epoch: int = None, best_val_loss: float = None,
                         stopped_early: bool = False):
    """Send a training complete notification."""
    parts = [f"**Training {'auto-stopped' if stopped_early else 'complete'}**"]
    parts.append(f"Total time: {total_time}")
    if best_epoch is not None:
        parts.append(f"Best epoch: {best_epoch} (val loss: `{best_val_loss:.4f}`)")
    send("\n".join(parts))
