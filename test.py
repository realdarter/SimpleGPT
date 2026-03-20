from chat_gen import *


if __name__ == "__main__":
    model_directory = 'checkpoint/run3'
    model, tokenizer = load_model_and_tokenizer(model_directory, download=False)
    args = create_args(
            max_length=512,
            temperature=0.8,
            top_k=60,
            top_p=0.92,
            repetition_penalty=1.2
        )

    while True:
        prompt_text = input("Input: ")
        response = generate_responses(model, tokenizer, prompt_text, args=args, clean_result=True)
        print(f"Prompt: {prompt_text}")
        print(f"Generated Response: {response}")