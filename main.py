import os
import contextlib

from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer, _get_device
)


def _reduce_host_contention():
    """Lower process priority and limit threads so the system doesn't lag."""
    import torch

    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(max(1, min(4, cpu_count // 2)))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                BELOW_NORMAL_PRIORITY_CLASS,
            )


if __name__ == "__main__":
    _reduce_host_contention()

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_SCRIPT_DIR, 'checkpoint', 'run')

    model, tokenizer = load_model_and_tokenizer(model_path, download=False)
    device = _get_device()

    args = create_args(
        max_length=512,
        max_new_tokens=64,
        max_newlines=2,
        temperature=0.65,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.22,
    )
    args["auto_correct_input"] = False

    print("Model loaded. Type 'quit' to exit.")
    while True:
        try:
            prompt_text = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if prompt_text.strip().lower() == 'quit':
            break
        response = generate_responses(model, tokenizer, prompt_text, device=device, args=args, clean_result=True)
        print(f"Bot: {response}")
