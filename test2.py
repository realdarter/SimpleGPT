from model_utils import *
import re
from tokenization import *

csv_path = "checkpoint/run2/cleaned.csv"
print(prepare_csv(csv_path))