# GPT-2 Chatbot Fine-Tuning 🤖
Fine-tune a pre-trained GPT-2 model to generate contextually coherent responses in a conversational setting. This guide will help you adapt GPT-2 for chat and reply tasks using your custom dataset.

# Requirements
- Python 3.6 or better
- Packages: torch transformers datasets (You can install these packages using `pip install torch transformers datasets`)

# Setup
- Open Terminal and navigate to your desired directory: {cd path_to_your_directory}
- Optional Setup separate Python environment {python -m venv myenv}
- Download the repository using `git clone [[your-repository-url]](https://github.com/realdarter/Goose-AI)`
- Prepare Dataset: Create a CSV file with context-response pairs. Ensure the file has a header with columns labeled context and reply.

# Features
- **Torch Debugging for GPU Use**: Leverages PyTorch’s capabilities for debugging and utilizing GPU acceleration.
- **Easy Access to Training**: Simple and customizable training configuration to suit various needs.

# Steps
1. **Install Dependencies**:
   ```bash
   pip install torch 
   pip install transformers
   pip install pandas
   ```
   Preferably install torch using NVIDIA GPU
   ```bash
   pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html```

2. **Prepare Dataset**:
   Create a dataset of context-response pairs for fine-tuning.
   Example of Context, Reply
   ![image](https://github.com/realdarter/Goose-AI/assets/100169417/7b65736c-4efd-430e-b408-b584d38a78cd)

   ```
   context,reply
   How are you doing so far?,I am doing fine!
   lol!,Thats funny
   "I need to talk to you","What, do, you, want"
   Food is an important part of our life, You are making me very hungry.
   ```

**Tokenize Dataset**: 
The script will automatically tokenize the dataset. Ensure your CSV file is correctly formatted and located in the specified path.

batch_size: Number of examples per training batch.
num_train_epochs: Number of training epochs.
learning_rate: Learning rate for optimizer.

**Train the Model**:
Run the script to start training. GPU acceleration will be used to speed up the process. The training progress will be printed in the terminal. The model will automatically save itself to the specified directory for future use to continue training.

# Example Code for Training and Testing:
# Train
```python
from model_utils import *
import os

model_path = 'checkpoint/cleaned.csv'
csv_path = os.path.join(model_path, 'cleaned.csv')

# Prepare the CSV data
args = create_training_args(
  num_epochs=3,
  batch_size=8,
  learning_rate=3e-5,
  save_every=1000,
  max_length=512,
  temperature=0.8,
  top_k=60,
  top_p=0.92,
  repetition_penalty=1.2
)

# Train the model (You can stop the model by exiting any time)
train_model(model_path, csv_path, args)
```python
from model_utils import *
model_directory = 'checkpoint/cleaned.csv'  # Replace with your actual model directory

prompt_text = input("Input: ")

args = create_training_args(
   num_epochs=3,
   batch_size=8,
   learning_rate=3e-5,
   save_every=1000,
   max_length=512,
   temperature=0.8,
   top_k=60,
   top_p=0.92,
   repetition_penalty=1.2
)
response = generate_responses(model_directory, prompt_text, args=args, clean_result=True)

print(f"Prompt: {prompt_text}")
print(f"Generated Response: {response}")
```



```plaintext
Fine-tuning GPT-2 involves preparing your dataset, configuring training parameters, and running the training script. The model will be saved for future use, and you can test it with new contexts to generate responses.
```
