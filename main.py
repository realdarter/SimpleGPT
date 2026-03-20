import re
import time
from chat_gen import (
    create_args, generate_responses, load_model_and_tokenizer,
    ensure_tokens, SPECIAL_TOKENS, _get_device
)
from discord import read_token, retrieve_user_data, retrieve_last_message, send_message


def replace_placeholders(text, userID):
    """Replace mentions starting with @ and [User] with <@userID>.
    Replace specific placeholders with predefined values.
    """
    text = re.sub(r'@[^ ]+', f'<@{userID}>', text)
    text = re.sub(r'\[User\]', f'<@{userID}>', text)
    text = re.sub(r'\[PROFANITY\]', '||[DICK]||', text)
    text = re.sub(r'\[EMAIL\]', 'dartiros.cc@gmail.com', text)
    text = re.sub(r'\[DISCORD\]', 'https://discord.gg/bugcat', text)
    text = re.sub(r'\[LINK\]', 'https://www.walmart.com/browse/personal-care/dildos/1005862_1414629_4054919_1811332', text)
    text = re.sub(r'\[PHONE\]', '505-503-4455', text)
    return text


if __name__ == "__main__":
    model_path = 'checkpoint/run3'
    channel_id = '898689047550132276'

    # Load model once at startup
    token = read_token()
    user_data = retrieve_user_data(token)
    if user_data is None:
        print("Failed to connect to Discord. Check your token and network.")
        exit(1)
    current_user = user_data['username']

    model, tokenizer = load_model_and_tokenizer(model_path, download=False)
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

    last_message_timestamp = 0

    while True:
        current_message = retrieve_last_message(token, channel_id)
        if (current_message is not None
                and current_message['timestamp'] != last_message_timestamp
                and current_message['content'] != ''
                and current_message['username'] != current_user):
            print(f"Found New Message: {current_message['content']}")
            trimmed_message = current_message['content'].strip()
            try:
                gen_response = generate_responses(model, tokenizer, trimmed_message, device=device, args=args, clean_result=True)
                reply_to_id = current_message.get('referenced_author_id') or current_message['author_id']
                filtered_message = replace_placeholders(gen_response, reply_to_id)
                send_message(token, channel_id, message="*[Dartiros AI]:* " + filtered_message, reply_to=current_message['message_id'])
                last_message_timestamp = current_message['timestamp']
            except Exception as e:
                print(f"Error processing message: {e}")

        time.sleep(1)
