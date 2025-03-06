from chat_gen import *


if __name__ == "__main__":
    model_directory = 'checkpoint/run2'  # Replace with your actual model directory
    model, tokenizer = load_model_and_tokenizer(model_directory, download=False)
    args = create_args(
            #num_epochs=3,
            #batch_size=4,
            #learning_rate=3e-5,
            #save_every=1000,
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