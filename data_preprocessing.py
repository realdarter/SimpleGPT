import pandas as pd
from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer

class ErrorReadingData(Exception):
    """Exception raised for errors encountered during data reading."""
    def __init__(self, message=None, file_path=None, file_type=None):
        self.message = message
        self.file_path = file_path
        self.file_type = file_type

    def __str__(self):
        if self.message and self.file_path and self.file_type:
            return f"Error Reading {self.file_type} file '{self.file_path}': {self.message}"
        elif self.message and self.file_path:
            return f"Error Reading file '{self.file_path}': {self.message}"
        elif self.message:
            return f"Error: {self.message}"
        else:
            return "ErrorReadingData has been raised"

class NoModelGiven(Exception):
    """Exception raised when no model is specified."""
    def __init__(self, message="No model specified"):
        self.message = message

    def __str__(self):
        return f"NoModelGiven: {self.message}"
    

def read_file(file_path):
    """
    Reads a file into a Pandas DataFrame based on file extension.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        pd.DataFrame: DataFrame containing the data read from the file.

    Raises:
        ErrorReadingData: If an error occurs during file reading.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            data = {'context': [], 'reply': []}
            for line in lines:
                context, reply = line.strip().split('\t')  # Assuming tab-separated values
                data['context'].append(context)
                data['reply'].append(reply)
            df = pd.DataFrame(data)
        else:
            raise ErrorReadingData("Unsupported file format", file_path, file_type=file_path.split('.')[-1])
        return df
    except Exception as e:
        raise ErrorReadingData(str(e), file_path, file_type=file_path.split('.')[-1])

def preprocess_data(df, model_name='gpt2'):
    """
    Tokenizes the data in the DataFrame using the specified tokenizer model.

    Args:
        df (pd.DataFrame): DataFrame containing 'context' and 'reply' columns.
        model_name (str, optional): Name of the tokenizer model to use. Default is 'gpt2'.

    Returns:
        pd.Series: Series containing tokenized data.

    Raises:
        NoModelGiven: If no model_name is specified.
        ValueError: If an unsupported model_name is provided or tokenization fails.
    """
    if not model_name:
        raise NoModelGiven("Please specify a model to use for tokenization.")
    
    # Initialize tokenizer based on specified model_name
    if model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            raise ValueError(f"Unsupported model '{model_name}'. Please choose 'gpt2' or 'bert', or a model compatible with AutoTokenizer.")
    
    # Tokenization function applied row-wise to DataFrame
    def tokenize_function(context, reply):
        return tokenizer(context + " " + tokenizer.eos_token + " " + reply, return_tensors='pt')

    try:
        tokenized_data = df.apply(lambda row: tokenize_function(row['context'], row['reply']), axis=1)
    except Exception as e:
        raise ValueError(f"Error tokenizing data: {str(e)}")

    return tokenized_data

if __name__ == "__main__":
    file_path = 'your_data.csv'  # Change to your file path
    try:
        df = read_file(file_path)
        tokenized_data = preprocess_data(df, model_name='gpt2')  # You can specify 'gpt2' or 'bert' here
        # Save or use the tokenized_data as needed
    except (ErrorReadingData, NoModelGiven, ValueError) as e:
        print(e)
