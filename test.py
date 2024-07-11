from model_utils import *
import re
from tokenization import *

model_path = 'checkpoint/run1'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

texts = "Hello I am 😀😃😄😁"
tokdata= tokenize_single_text(tokenizer, texts, max_length=512)
token_ids = tokdata['input_ids'][0]
print(token_ids)

# Ensure tokdata[1] is a flat list of token IDs
decodeddata = decode_data(tokenizer, token_ids)
print(decodeddata)
