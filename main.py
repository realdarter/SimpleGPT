from model_utils import *
import re
from discord import *

def replace_placeholders(text, userID):
    """Replace mentions starting with @ and [User] with <@userID>.
    Replace specific placeholders with predefined values.
    """
    pattern_mentions = r'@[^ ]+'
    pattern_user = r'\[User\]'
    pattern_profanity = r'\[PROFANITY\]'
    pattern_email = r'\[EMAIL\]'
    pattern_discord = r'\[DISCORD\]'
    pattern_link = r'\[LINK\]'
    pattern_phone = r'\[PHONE\]'

    # Replace mentions and [User] with <@userID>
    text = re.sub(pattern_mentions, f'<@{userID}>', text)
    text = re.sub(pattern_user, f'<@{userID}>', text)

    # Replace placeholders with specific values
    text = re.sub(pattern_profanity, '||[DICK]||', text)
    text = re.sub(pattern_email, 'dartiros.cc@gmail.com', text)
    text = re.sub(pattern_discord, 'https://discord.gg/bugcat', text)
    text = re.sub(pattern_link, 'https://www.walmart.com/browse/personal-care/dildos/1005862_1414629_4054919_1811332', text)
    text = re.sub(pattern_phone, '505-503-4455', text)

    return text

if __name__ == "__main__":
    model_path = 'checkpoint/run1'
    channel_id = '898689047550132276'
    
    global last_message_timestamp
    last_message_timestamp = 0
    token = read_token()
    user_data = retrieve_user_data(token)
    current_user = user_data['username']

    retrieve_user_data(read_token())

    while True:
        #test_prompt = input("Input: ")
        #test_prompt = f"<|startoftext|> {test_prompt} <|septext|>"
        #print(generate_responses(model_path, test_prompt))
        current_message = retrieve_last_message(token, channel_id)
        #if ((current_message is not None) and (current_message['timestamp'] != last_message_timestamp) and (current_message['content'] != '') and (current_message['username'] == "Beef Bot")):
        if ((current_message is not None) and (current_message['timestamp'] != last_message_timestamp) and (current_message['content'] != '') and (current_message['username'] != current_user)):
            print(f"Found New Message: {current_message['content']}")
            trimmed_message = f"{current_message['content'].strip()}"
            #trimmed_message = f"{current_message['content'].strip()}"
            try:
                args = create_training_args(
                    num_epochs=3,
                    batch_size=4,
                    learning_rate=3e-5,
                    save_every=1000,
                    max_length=512,
                    temperature=0.8,
                    top_k=60,
                    top_p=0.92,
                    repetition_penalty=1.2
                )
                gen_response = generate_responses(model_path, trimmed_message, args=args, clean_result=True)
                filtered_message = replace_placeholders(gen_response, current_message['referenced_author_id'])
                send_message(read_token(), channel_id, message="*[Dartiros AI]:* " + filtered_message, reply_to=current_message['message_id'])
                #send_message(read_token(), channel_id, message=filtered_message, reply_to=current_message['message_id'])
                last_message_timestamp = current_message['timestamp']
            except Exception as e:
                print(f"Error processing message: {e}")

        time.sleep(1)