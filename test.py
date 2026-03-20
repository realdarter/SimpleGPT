from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer,
    ensure_tokens, SPECIAL_TOKENS, _get_device
)

if __name__ == "__main__":
    model_directory = 'checkpoint/run3'
    model, tokenizer = load_model_and_tokenizer(model_directory, download=False)
    device = _get_device()
    ensure_tokens(model, tokenizer, special_tokens=SPECIAL_TOKENS)
    model.to(device)

    args = create_args(
        max_length=512,
        max_new_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2
    )

    while True:
        prompt_text = input("Input: ")
        response = generate_responses(model, tokenizer, prompt_text, device=device, args=args, clean_result=True)
        print(f"Prompt: {prompt_text}")
        print(f"Generated Response: {response}")
