"""
Dara Discord bot for SimpleGPT training and inference.

This version focuses on reliability:
- Centralized runtime state instead of scattered globals
- Serialized model load / reload / inference
- Safer background training subprocess handling
- Cleaner Discord-safe output chunking and error reporting
- Configurable token / channel / model paths via env vars
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import os
import subprocess
import sys
import time
import traceback
import ctypes
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import discord
from discord import app_commands

from chat_gen import (
    _get_device,
    create_args,
    generate_responses,
    load_model_and_tokenizer,
)


DISCORD_MESSAGE_LIMIT = 2000
CODE_BLOCK_LIMIT = 1900
TRAIN_LOG_INTERVAL_SEC = 8.0
TRAIN_LOG_LINE_WINDOW = 30
IMPORTANT_TRAIN_TOKENS = (
    "Epoch",
    "Checkpoint",
    "Auto-stopped",
    "Training completed",
    "Error",
    "OOM",
    "Traceback",
)
STREAM_SENTINEL = object()
MODEL_IDLE_TIMEOUT_SEC = 1200  # 20 mins — unload model if no /ask in this time


def _compute_model_hash(model_dir: str) -> str:
    """Hash the adapter + config files to detect model changes (e.g. after training)."""
    h = hashlib.sha256()
    adapter_dir = os.path.join(model_dir, "lora_adapter")
    base_dir = os.path.join(model_dir, "base_model")

    # Hash adapter files if they exist, otherwise hash base config
    target_dir = adapter_dir if os.path.isdir(adapter_dir) else base_dir
    if not os.path.isdir(target_dir):
        return "no_model"

    for fname in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Hash weights + config, skip huge files byte-by-byte — use size+mtime as proxy
        if fname.endswith((".safetensors", ".bin")):
            stat = os.stat(fpath)
            h.update(f"{fname}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        elif fname.endswith((".json", ".txt", ".model")):
            with open(fpath, "rb") as f:
                h.update(f.read())

    return h.hexdigest()[:16]


@dataclass(frozen=True)
class BotConfig:
    channel_id: int
    model_dir: str
    script_dir: str
    train_script: str


@dataclass
class RuntimeState:
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    device: Optional[Any] = None
    gen_args: Optional[dict[str, Any]] = None
    model_loaded_at: Optional[float] = None
    model_hash: Optional[str] = None
    last_ask_at: Optional[float] = None
    idle_task: Optional[asyncio.Task] = None

    train_process: Optional[subprocess.Popen[str]] = None
    train_stream_task: Optional[asyncio.Task] = None
    training_started_at: Optional[float] = None
    training_channel_id: Optional[int] = None
    train_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=200))
    train_status_msg: Optional[Any] = None  # Discord message to edit in-place

    model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    train_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = BotConfig(
    channel_id=int(os.getenv("DISCORD_CHANNEL_ID", "1346717893911511051")),
    model_dir=os.getenv("SIMPLEGPT_MODEL_DIR", "checkpoint/run"),
    script_dir=SCRIPT_DIR,
    train_script=os.path.join(SCRIPT_DIR, "train.py"),
)
STATE = RuntimeState()


intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


def _get_python_exe() -> str:
    return sys.executable


def _is_training() -> bool:
    return STATE.train_process is not None and STATE.train_process.poll() is None


def _format_duration(started_at: Optional[float]) -> str:
    if not started_at:
        return "n/a"
    seconds = max(0, int(time.time() - started_at))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_train_excerpt(lines: list[str]) -> str:
    excerpt = "\n".join(lines[-TRAIN_LOG_LINE_WINDOW:])
    if len(excerpt) > CODE_BLOCK_LIMIT:
        excerpt = "...\n" + excerpt[-(CODE_BLOCK_LIMIT - 4) :]
    return excerpt


def _escape_code_block(text: str) -> str:
    return text.replace("```", "'''")


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks


async def _send_text_chunks(target, text: str, *, code_block: bool = False, prefix: str = "") -> None:
    payload = _escape_code_block(text)
    chunks = _chunk_text(payload, CODE_BLOCK_LIMIT if code_block else DISCORD_MESSAGE_LIMIT - len(prefix))

    for index, chunk in enumerate(chunks):
        message_prefix = prefix if index == 0 else ""
        if code_block:
            content = f"{message_prefix}```\n{chunk}\n```"
        else:
            content = message_prefix + chunk
        await target.send(content)


async def _safe_interaction_reply(
    interaction: discord.Interaction,
    text: str,
    *,
    code_block: bool = False,
    prefix: str = "",
) -> None:
    target = interaction.followup if interaction.response.is_done() else interaction.response
    payload = _escape_code_block(text)
    limit = CODE_BLOCK_LIMIT if code_block else DISCORD_MESSAGE_LIMIT - len(prefix)
    chunks = _chunk_text(payload, limit)

    for index, chunk in enumerate(chunks):
        message_prefix = prefix if index == 0 else ""
        if code_block:
            content = f"{message_prefix}```\n{chunk}\n```"
        else:
            content = message_prefix + chunk

        if target is interaction.response:
            await interaction.response.send_message(content)
            target = interaction.followup
        else:
            await interaction.followup.send(content)


async def _resolve_channel(channel_id: int):
    channel = client.get_channel(channel_id)
    if channel is not None:
        return channel

    with contextlib.suppress(discord.HTTPException, discord.Forbidden, discord.NotFound):
        return await client.fetch_channel(channel_id)
    return None


def _gpu_status_text() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CPU only"

    gpu_name = torch.cuda.get_device_name(0)
    mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f"{gpu_name} ({mem_used:.1f} GB allocated, {mem_reserved:.1f} GB reserved / {mem_total:.1f} GB total)"


def _reduce_host_contention() -> None:
    """Keep the bot from monopolizing the whole machine during load/inference."""
    try:
        import torch

        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(max(1, min(4, cpu_count // 2)))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    if os.name == "nt":
        with contextlib.suppress(Exception):
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                BELOW_NORMAL_PRIORITY_CLASS,
            )


def _load_model_blocking() -> str:
    _reduce_host_contention()
    current_hash = _compute_model_hash(CONFIG.model_dir)

    # Already loaded and unchanged — skip
    if STATE.model is not None and STATE.model_hash == current_hash:
        return f"Model already loaded on `{STATE.device}` (hash `{current_hash}`)."

    # Hash changed — unload old model first
    if STATE.model is not None and STATE.model_hash != current_hash:
        print(f"[Model] Hash changed: {STATE.model_hash} -> {current_hash}, reloading...")
        _unload_model_blocking()

    try:
        model, tokenizer = load_model_and_tokenizer(CONFIG.model_dir, download=False)
    except FileNotFoundError as exc:
        base_model_dir = os.path.join(CONFIG.model_dir, "base_model")
        if os.path.isdir(base_model_dir):
            raise
        print(f"[Model] Base model missing at {base_model_dir}, attempting download...")
        try:
            model, tokenizer = load_model_and_tokenizer(CONFIG.model_dir, download=True)
        except Exception as download_exc:
            raise FileNotFoundError(
                f"Base model missing at {base_model_dir} and automatic download failed. "
                f"Restore the model files there or let the bot download them. "
                f"Original error: {exc}. Download error: {download_exc}"
            ) from download_exc
    device = _get_device()

    try:
        model.to(device)
    except Exception:
        pass

    current_hash = _compute_model_hash(CONFIG.model_dir)

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.device = device
    STATE.gen_args = create_args(
        max_length=512,
        max_new_tokens=48,
        max_newlines=1,
        temperature=0.35,
        top_k=24,
        top_p=0.82,
        repetition_penalty=1.12,
    )
    STATE.gen_args["auto_correct_input"] = False
    STATE.model_loaded_at = time.time()
    STATE.model_hash = current_hash
    return f"Model loaded on `{device}` (hash `{current_hash}`)."


def _unload_model_blocking() -> str:
    import torch

    if STATE.model is not None:
        del STATE.model
    if STATE.tokenizer is not None:
        del STATE.tokenizer

    STATE.model = None
    STATE.tokenizer = None
    STATE.device = None
    STATE.gen_args = None
    STATE.model_loaded_at = None
    STATE.model_hash = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "Model unloaded."


async def _ensure_model_loaded() -> str:
    async with STATE.model_lock:
        result = await asyncio.to_thread(_load_model_blocking)
        _reset_idle_timer()
        return result


async def _reload_model() -> str:
    async with STATE.model_lock:
        await asyncio.to_thread(_unload_model_blocking)
        result = await asyncio.to_thread(_load_model_blocking)
        _reset_idle_timer()
        return result


async def _unload_model() -> str:
    async with STATE.model_lock:
        _cancel_idle_timer()
        return await asyncio.to_thread(_unload_model_blocking)


def _reset_idle_timer() -> None:
    """Reset the 1-hour idle countdown. If no /ask within the timeout, unload the model."""
    STATE.last_ask_at = time.time()
    if STATE.idle_task is not None:
        STATE.idle_task.cancel()
    STATE.idle_task = asyncio.ensure_future(_idle_unload_watcher())


def _cancel_idle_timer() -> None:
    if STATE.idle_task is not None:
        STATE.idle_task.cancel()
        STATE.idle_task = None
    STATE.last_ask_at = None


async def _idle_unload_watcher() -> None:
    """Background task: unloads the model after MODEL_IDLE_TIMEOUT_SEC of inactivity."""
    try:
        while True:
            await asyncio.sleep(60)
            if STATE.model is None or STATE.last_ask_at is None:
                return
            if _is_training():
                continue
            idle_sec = time.time() - STATE.last_ask_at
            if idle_sec >= MODEL_IDLE_TIMEOUT_SEC:
                print(f"[Idle] No /ask for {idle_sec:.0f}s — unloading model to save resources.")
                async with STATE.model_lock:
                    await asyncio.to_thread(_unload_model_blocking)
                channel = await _resolve_channel(CONFIG.channel_id)
                if channel is not None:
                    with contextlib.suppress(discord.HTTPException):
                        await channel.send("Model auto-unloaded after 20 min idle. Next `/ask` will reload it.")
                return
    except asyncio.CancelledError:
        return
    except Exception as e:
        print(f"[Idle Watcher Error] {e}")
        traceback.print_exc()


async def _generate_response(prompt: str) -> str:
    async with STATE.inference_lock:
        await _ensure_model_loaded()
        await asyncio.to_thread(_reduce_host_contention)
        return await asyncio.to_thread(
            generate_responses,
            STATE.model,
            STATE.tokenizer,
            prompt,
            device=STATE.device,
            args=STATE.gen_args,
            clean_result=True,
        )


def _build_train_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _pump_process_output(process: subprocess.Popen[str], loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    try:
        if process.stdout is None:
            return

        for line in iter(process.stdout.readline, ""):
            loop.call_soon_threadsafe(queue.put_nowait, line.rstrip())
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, STREAM_SENTINEL)


async def _update_train_status(channel, lines: list[str]) -> None:
    """Edit the single training status message. Never sends new messages."""
    if channel is None or not lines:
        return

    excerpt = _format_train_excerpt(lines)
    if not excerpt.strip():
        return

    content = f"```\n{_escape_code_block(excerpt)}\n```"
    if len(content) > DISCORD_MESSAGE_LIMIT:
        content = content[:DISCORD_MESSAGE_LIMIT - 3] + "```"

    try:
        if STATE.train_status_msg is not None:
            await STATE.train_status_msg.edit(content=content)
        else:
            STATE.train_status_msg = await channel.send(content)
    except (discord.NotFound, discord.HTTPException):
        with contextlib.suppress(discord.HTTPException):
            STATE.train_status_msg = await channel.send(content)


async def _stream_train_output(process: subprocess.Popen[str], channel_id: int) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    pump_task = asyncio.create_task(asyncio.to_thread(_pump_process_output, process, loop, queue))
    channel = await _resolve_channel(channel_id)
    pending_lines: list[str] = []
    last_flush = time.monotonic()
    exit_code: Optional[int] = None
    final_tail: list[str] = []

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=TRAIN_LOG_INTERVAL_SEC)
            except asyncio.TimeoutError:
                item = None

            if item is STREAM_SENTINEL:
                break

            if isinstance(item, str) and item:
                pending_lines.append(item)
                STATE.train_tail.append(item)
                print(item)

            # Always edit the same message
            if pending_lines and (time.monotonic() - last_flush) >= TRAIN_LOG_INTERVAL_SEC:
                await _update_train_status(channel, pending_lines)
                pending_lines.clear()
                last_flush = time.monotonic()

        if pending_lines:
            await _update_train_status(channel, pending_lines)

        with contextlib.suppress(subprocess.TimeoutExpired):
            await asyncio.to_thread(process.wait, timeout=2)
        exit_code = process.poll()
        final_tail = list(STATE.train_tail)

    finally:
        await pump_task
        async with STATE.train_lock:
            if STATE.train_process is process:
                STATE.train_process = None
                STATE.train_stream_task = None
                STATE.training_started_at = None
                STATE.training_channel_id = None

        # Final message — edit the status msg one last time
        if channel is not None:
            summary = (
                f"**Training finished** with exit code `{exit_code}`."
                if exit_code is not None
                else "**Training finished.**"
            )
            try:
                if STATE.train_status_msg is not None:
                    await STATE.train_status_msg.edit(content=summary)
                else:
                    await channel.send(summary)
            except (discord.NotFound, discord.HTTPException):
                with contextlib.suppress(discord.HTTPException):
                    await channel.send(summary)
            STATE.train_status_msg = None


async def _start_training(channel_id: int) -> tuple[str, bool]:
    async with STATE.train_lock:
        if _is_training():
            raise RuntimeError("Training is already running.")

        unloaded_model = False
        if STATE.model is not None:
            async with STATE.inference_lock:
                await _unload_model()
                unloaded_model = True

        python_exe = _get_python_exe()
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            [python_exe, CONFIG.train_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=CONFIG.script_dir,
            env=_build_train_environment(),
            creationflags=creationflags,
        )

        STATE.train_process = process
        STATE.training_started_at = time.time()
        STATE.training_channel_id = channel_id
        STATE.train_tail.clear()
        STATE.train_stream_task = asyncio.create_task(_stream_train_output(process, channel_id))

        return f"Training started with PID `{process.pid}`.", unloaded_model


async def _stop_training() -> str:
    async with STATE.train_lock:
        process = STATE.train_process
        stream_task = STATE.train_stream_task

    if process is None or process.poll() is not None:
        async with STATE.train_lock:
            STATE.train_process = None
            STATE.train_stream_task = None
            STATE.training_started_at = None
            STATE.training_channel_id = None
        return "Nothing is currently running."

    with contextlib.suppress(Exception):
        process.terminate()

    try:
        await asyncio.to_thread(process.wait, timeout=8)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(process.wait, timeout=3)

    if stream_task is not None:
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(asyncio.shield(stream_task), timeout=10)

    return "Training process stopped."


def _format_exception(exc: BaseException) -> str:
    """Format exception with full traceback for debugging."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Truncate if too long for Discord
    if len(tb) > 1800:
        tb = tb[:900] + "\n...\n" + tb[-900:]
    return tb


@tree.command(name="ask", description="Ask the model a question")
@app_commands.describe(question="Your question or prompt for the model")
async def cmd_ask(interaction: discord.Interaction, question: str) -> None:
    if _is_training():
        await interaction.response.send_message(
            "Training is running, so generation is disabled until it finishes or you use `/stop`."
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        # Check hash — auto-reload if model changed (e.g. after training)
        load_note = None
        current_hash = _compute_model_hash(CONFIG.model_dir)
        if STATE.model is None or STATE.model_hash != current_hash:
            action = "Reloading (model updated)" if STATE.model is not None else "Loading model"
            load_note = await _ensure_model_loaded()
            load_note = f"{action}... {load_note}"

        # Track last usage for idle timeout
        STATE.last_ask_at = time.time()
        _reset_idle_timer()

        response = await _generate_response(question)
        response = response.strip() or "(empty response)"
        response = _truncate(response, 3500)

        if load_note:
            await interaction.followup.send(load_note)
        await _safe_interaction_reply(
            interaction,
            f"**Q:** {question}\n**A:** {response}",
        )
    except Exception as exc:
        await _safe_interaction_reply(interaction, _format_exception(exc), code_block=True, prefix="Error:\n")


@tree.command(name="test", description="Run a test generation with logs")
@app_commands.describe(prompt="The prompt to test with")
async def cmd_test(interaction: discord.Interaction, prompt: str = "Hello, how are you?") -> None:
    if _is_training():
        await interaction.response.send_message(
            "Training is running, so test generation is disabled until it finishes or you use `/stop`."
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        logs: list[str] = []
        if STATE.model is None:
            logs.append(await _ensure_model_loaded())

        logs.append(f"Device: {STATE.device}")
        logs.append(f"GPU: {_gpu_status_text()}")
        logs.append(f"Prompt: {prompt}")
        logs.append("Generating...")

        started_at = time.time()
        response = await _generate_response(prompt)
        logs.append(f"Elapsed: {time.time() - started_at:.2f}s")
        logs.append(f"Response: {response.strip() or '(empty response)'}")

        await _safe_interaction_reply(interaction, "\n".join(logs), code_block=True, prefix="Test Results:\n")
    except Exception as exc:
        await _safe_interaction_reply(interaction, _format_exception(exc), code_block=True, prefix="Error:\n")


@tree.command(name="train", description="Start model training in the background")
async def cmd_train(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)

    try:
        channel_id = interaction.channel_id or CONFIG.channel_id
        message, unloaded_model = await _start_training(channel_id)
        suffix = "\nModel was unloaded first to free GPU memory." if unloaded_model else ""
        await interaction.followup.send(f"{message}\nProgress will stream to this channel.{suffix}")
    except Exception as exc:
        await _safe_interaction_reply(interaction, _format_exception(exc), code_block=True, prefix="Error starting training:\n")


@tree.command(name="stop", description="Stop the running training process")
async def cmd_stop(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    message = await _stop_training()
    await interaction.followup.send(message)


@tree.command(name="status", description="Check model, GPU, and training status")
async def cmd_status(interaction: discord.Interaction) -> None:
    parts = []

    if STATE.model is not None:
        idle_str = ""
        if STATE.last_ask_at:
            idle_sec = int(time.time() - STATE.last_ask_at)
            remaining = max(0, MODEL_IDLE_TIMEOUT_SEC - idle_sec)
            idle_str = f" | idle {idle_sec}s, auto-unload in {remaining // 60}m"
        parts.append(f"**Model:** Loaded on `{STATE.device}` for {_format_duration(STATE.model_loaded_at)} (hash `{STATE.model_hash}`{idle_str})")
    else:
        disk_hash = _compute_model_hash(CONFIG.model_dir)
        parts.append(f"**Model:** Not loaded (disk hash `{disk_hash}`)")

    if _is_training():
        process = STATE.train_process
        pid_text = f" (PID {process.pid})" if process is not None else ""
        parts.append(f"**Training:** Running for {_format_duration(STATE.training_started_at)}{pid_text}")
    else:
        parts.append("**Training:** Idle")

    parts.append(f"**GPU:** {_gpu_status_text()}")
    parts.append(f"**Log Channel:** <#{STATE.training_channel_id or CONFIG.channel_id}>")

    if STATE.train_tail:
        parts.append(f"**Last Train Line:** `{_truncate(STATE.train_tail[-1], 120)}`")

    await interaction.response.send_message("\n".join(parts))


@tree.command(name="reload", description="Unload and reload the model from disk")
async def cmd_reload(interaction: discord.Interaction) -> None:
    if _is_training():
        await interaction.response.send_message(
            "Training is running, so reload is disabled until it finishes or you use `/stop`."
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        async with STATE.inference_lock:
            message = await _reload_model()
        await interaction.followup.send(f"Model reloaded. {message}")
    except Exception as exc:
        await _safe_interaction_reply(interaction, _format_exception(exc), code_block=True, prefix="Error reloading:\n")


@tree.command(name="unload", description="Unload the model from GPU/CPU memory")
async def cmd_unload(interaction: discord.Interaction) -> None:
    if _is_training():
        await interaction.response.send_message(
            "Training is running, so unload is disabled until it finishes or you use `/stop`."
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        async with STATE.inference_lock:
            message = await _unload_model()
        await interaction.followup.send(message)
    except Exception as exc:
        await _safe_interaction_reply(interaction, _format_exception(exc), code_block=True, prefix="Error unloading:\n")


@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    original = getattr(error, "original", error)
    message = _format_exception(original)
    print(f"[Command Error] {message}")

    try:
        content = f"Command failed:\n```\n{_escape_code_block(message)}\n```"
        if len(content) > DISCORD_MESSAGE_LIMIT:
            content = content[:DISCORD_MESSAGE_LIMIT - 3] + "```"
        if interaction.response.is_done():
            await interaction.followup.send(content)
        else:
            await interaction.response.send_message(content)
    except Exception as e:
        print(f"[Error Handler Failed] Could not send error to Discord: {e}")


@client.event
async def on_error(event: str, *args, **kwargs) -> None:
    """Catch any unhandled error so the bot never crashes."""
    print(f"[Bot Error] Unhandled error in {event}:")
    traceback.print_exc()


_synced = False

@client.event
async def on_ready() -> None:
    global _synced
    if not _synced:
        await tree.sync()
        _synced = True
        print("Slash commands synced: /ask /test /train /stop /status /reload /unload")
    print(f"Bot ready as {client.user} (ID: {client.user.id})")

    channel = await _resolve_channel(CONFIG.channel_id)
    if channel is not None:
        with contextlib.suppress(discord.HTTPException):
            await channel.send(
                "**Dara is online.**\n"
                "`/ask` ask the model\n"
                "`/test` run a logged generation\n"
                "`/train` start training\n"
                "`/stop` stop training\n"
                "`/status` inspect model and GPU state\n"
                "`/reload` reload the current checkpoint\n"
                "`/unload` free model memory"
            )


def _load_token() -> Optional[str]:
    env_token = os.getenv("DISCORD_BOT_TOKEN")
    if env_token:
        return env_token.strip()

    auth_path = os.path.join(CONFIG.script_dir, "auth.txt")
    if not os.path.isfile(auth_path):
        return None

    with open(auth_path, "r", encoding="utf-8") as handle:
        token = handle.read().strip()
    return token or None


def main() -> None:
    token = _load_token()
    if not token or token == "PASTE_YOUR_BOT_TOKEN_HERE":
        print("Error: set DISCORD_BOT_TOKEN or put a valid token in auth.txt.")
        return

    print(f"[Output Processing Location] Discord Channel: {CONFIG.channel_id}")
    print(f"[Model Directory] {CONFIG.model_dir}")

    while True:
        try:
            client.run(token)
            break  # clean exit
        except KeyboardInterrupt:
            print("\n[Bot] Shutting down...")
            break
        except Exception as e:
            print(f"\n[Bot Crash] {e}")
            traceback.print_exc()
            print("[Bot] Restarting in 10 seconds...")
            time.sleep(10)


if __name__ == "__main__":
    main()
