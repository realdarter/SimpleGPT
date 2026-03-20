import requests
import json
import datetime
import time

API_VERSION = "v9"
API_BASE = f"https://discord.com/api/{API_VERSION}"
REQUEST_TIMEOUT = 10  # seconds


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
        'Referer': f'https://discord.com/channels/@me/{channel_id}',
    }

    r = requests.post(
        f'{API_BASE}/channels/{channel_id}/messages',
        headers=headers, json=payload, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def retrieve_user_data(token):
    headers = {'Authorization': token}
    url = f'{API_BASE}/users/@me'
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        user_data = {
            'id': data['id'],
            'username': data['username'],
            'avatar': data['avatar'],
            'discriminator': data['discriminator'],
            'public_flags': data['public_flags'],
            'flags': data['flags'],
            'global_name': data.get('global_name', ''),
            'banner': data.get('banner', None),
            'accent_color': data.get('accent_color', None),
            'clan': data.get('clan', None),
            'mfa_enabled': data['mfa_enabled'],
            'locale': data['locale'],
            'premium_type': data['premium_type'],
            'email': data.get('email', ''),
            'verified': data['verified'],
            'phone': data.get('phone', None),
            'nsfw_allowed': data.get('nsfw_allowed', False),
            'linked_users': data['linked_users'],
            'bio': data.get('bio', ''),
            'authenticator_types': data['authenticator_types']
        }

        return user_data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return None


def retrieve_messages(token, channel_id, number=1):
    headers = {'Authorization': token}
    url = f'{API_BASE}/channels/{channel_id}/messages?limit={number}'

    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        messages_list = []
        for message in data:
            referenced = message.get('referenced_message')

            messages_list.append({
                'message_id': message['id'],
                'author_id': message['author']['id'],
                'username': message['author']['username'],
                'avatar': message['author']['avatar'],
                'timestamp': message['timestamp'],
                'content': message['content'],
                'referenced_author_id': referenced['author']['id'] if referenced else None,
                'referenced_author_username': referenced['author']['username'] if referenced else None,
            })

        return messages_list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching messages: {e}")
        return None


def retrieve_last_message(token, channel_id):
    messages = retrieve_messages(token, channel_id, 1)
    if messages:
        return messages[0]
    return None
