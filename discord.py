import requests
import json
import datetime
import time

# Read the token from the file and strip any surrounding whitespace
def read_token():
    with open("token.txt", "r") as file:
        token = file.read().strip()
    return token

def nonce():
    date = datetime.datetime.now()
    unixts = time.mktime(date.timetuple())
    return str((int(unixts) * 1000 - 1420070400000) * 4194304)

def send_message(token, channel_id, message, tts=False, embed=None, allowed_mentions=None, components=None, reply_to=None):
    payload = {
        'content': message,
        'nonce': nonce(),
        'tts': tts,
        'embed': embed,
        'allowed_mentions': allowed_mentions,
        'components': components
    }

    if reply_to:
        payload['message_reference'] = {
            'channel_id': channel_id,
            'message_id': reply_to
        }

    headers = {
        'Authorization': token,
        #'Accept': '*/*',
        #'Accept-Encoding': 'gzip, deflate, br, zstd',
        #'Accept-Language': 'en-US,en;q=0.9',
        #'Content-Type': 'application/json',
        #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        #'Origin': 'https://discord.com',
        'Referer': f'https://discord.com/channels/@me/{channel_id}',
        #'Sec-Ch-Ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        #'Sec-Ch-Ua-Mobile': '?0',
        #'Sec-Ch-Ua-Platform': '"Windows"',
        #'Sec-Fetch-Dest': 'empty',
        #'Sec-Fetch-Mode': 'cors',
        #'Sec-Fetch-Site': 'same-origin',
        #'X-Debug-Options': 'bugReporterEnabled',
        #'X-Discord-Locale': 'en-US',
        #'X-Discord-Timezone': 'America/Los_Angeles',
        #'X-Super-Properties': 
    }

    r = requests.post(f'https://discord.com/api/v9/channels/{channel_id}/messages', 
                      headers=headers, json=payload)
    #print(r.status_code, r.text)


def retrieve_user_data(token):
    headers = {'Authorization': token}
    url = 'https://discord.com/api/v9/users/@me'
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()  # Raise an exception for bad status codes
        jsonn = json.loads(r.text)
        #print(jsonn)
        # Extract relevant data
        user_data = {
            'id': jsonn['id'],
            'username': jsonn['username'],
            'avatar': jsonn['avatar'],
            'discriminator': jsonn['discriminator'],
            'public_flags': jsonn['public_flags'],
            'flags': jsonn['flags'],
            'global_name': jsonn.get('global_name', ''),
            'banner': jsonn.get('banner', None),
            'accent_color': jsonn.get('accent_color', None),
            'clan': jsonn.get('clan', None),
            'mfa_enabled': jsonn['mfa_enabled'],
            'locale': jsonn['locale'],
            'premium_type': jsonn['premium_type'],
            'email': jsonn.get('email', ''),
            'verified': jsonn['verified'],
            'phone': jsonn.get('phone', None),
            'nsfw_allowed': jsonn.get('nsfw_allowed', False),
            'linked_users': jsonn['linked_users'],
            'bio': jsonn.get('bio', ''),
            'authenticator_types': jsonn['authenticator_types']
        }
        
        return user_data
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return None

def retrieve_messages(token, channel_id, number=1):
    headers = {'Authorization': token}
    url = f'https://discord.com/api/v8/channels/{channel_id}/messages?limit={number}'
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        jsonn = json.loads(r.text)
        #print(jsonn)
        messages_list = []
        #print(jsonn)
        for message in jsonn:
            referenced_message = message.get('referenced_message')
            
            

            
            messages_list.append({
                'message_id': message['id'],
                'username': message['author']['username'],
                'timestamp': message['timestamp'],
                'content': message['content'],
                'referenced_author_id': message['author']['id'],
                'referenced_author_username': message['author']['username'],
                'referenced_author_avatar': message['author']['avatar'],
            })
        
        return messages_list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching messages: {e}")
        return None

"""
def retrieve_messages(token, channel_id, number=1):
    headers = {'Authorization': token}
    messages_list = []
    last_message_id = None

    try:
        while number > 0:
            limit = min(number, 100)
            url = f'https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}'
            
            if last_message_id:
                url += f'&before={last_message_id}'

            r = requests.get(url, headers=headers)
            r.raise_for_status()
            jsonn = r.json()

            if not jsonn:
                break
            
            messages_list.extend([
                {
                    'username': message['author']['username'],
                    'timestamp': message['timestamp'],
                    'content': message['content']
                }
                for message in jsonn
            ])

            number -= len(jsonn)
            last_message_id = jsonn[-1]['id']

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return messages_list
"""

def retrieve_last_message(token, channel_id):
    messages = retrieve_messages(token, channel_id, 1)
    if messages:
        return messages[0]
    return None


# Example usage
#channel_id = '593267212266242049'
#last_message_timestamp = None
#retrieve_user_data(read_token())
#print(retrieve_last_message(read_token(), channel_id))
#send_message(read_token(), channel_id, message="<@368874228855865354>")
#last_message = retrieve_last_message(read_token(), channel_id)
#if last_message:
    #send_message(read_token(), channel_id, message="HI GUYS", reply_to=last_message['message_id'])

