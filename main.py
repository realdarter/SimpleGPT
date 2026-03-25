from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer, _get_device
)

if __name__ == "__main__":
    model_path = 'checkpoint/run'

    model, tokenizer = load_model_and_tokenizer(model_path, download=False)
    device = _get_device()

    args = create_args(
        max_length=512,
        max_new_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.92,
        repetition_penalty=1.2
    )

    print("Model loaded. Type 'quit' to exit.")
    while True:
        prompt_text = input("\nYou: ")
        if prompt_text.strip().lower() == 'quit':
            break
        response = generate_responses(model, tokenizer, prompt_text, device=device, args=args, clean_result=True)
        print(f"Bot: {response}")
