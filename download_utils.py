import os
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def download_file_with_progress(url_base, sub_dir, model_name, file_name):
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    try:
        r = requests.get(url_base + "/models/" + model_name + "/" + file_name, stream=True)
        with open(os.path.join(sub_dir, file_name), 'wb') as f:
            file_size = int(r.headers["content-length"])
            with tqdm(desc="Fetching " + file_name, total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        logging.error(f"Error downloading file '{file_name}' from {url_base}: {str(e)}")
        raise

def is_gpt2_downloaded(model_dir='models', model_name='124M'):
    """
    Check if GPT-2 model files are already downloaded.
    """
    sub_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(sub_dir):
        return False
    
    required_files = ['config.json', 'merges.txt', 'pytorch_model.bin', 'tokenizer.json', 'vocab.json']
    
    # Check if all required files exist
    for file_name in required_files:
        if not os.path.exists(os.path.join(sub_dir, file_name)):
            return False
    
    return True

def download_gpt2(model_dir='models', model_name='124M'):
    try:
        sub_dir = os.path.join(model_dir, model_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        sub_dir = sub_dir.replace('\\', '/')  # Windows compatibility

        for file_name in ['config.json', 'merges.txt', 'pytorch_model.bin', 'tokenizer.json', 'vocab.json']:
            download_file_with_progress(url_base="https://openaipublic.blob.core.windows.net/gpt-2",
                                        sub_dir=sub_dir,
                                        model_name=model_name,
                                        file_name=file_name)
    except Exception as e:
        logging.error(f"Error downloading GPT-2 model '{model_name}': {str(e)}")
        raise
