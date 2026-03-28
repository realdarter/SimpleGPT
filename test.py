from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer, _get_device
)

if __name__ == "__main__":
    model_directory = 'checkpoint/run'
    model, tokenizer = load_model_and_tokenizer(model_directory, download=False)
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

    while True:
        prompt_text = input("Input: ")
        response = generate_responses(model, tokenizer, prompt_text, device=device, args=args, clean_result=True)
        print(f"Prompt: {prompt_text}")
        print(f"Generated Response: {response}")
