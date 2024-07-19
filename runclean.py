import pandas as pd
import os

# ANSI escape codes for colors
class colors:
    BLUE = '\033[99m'  # Whiter
    GREEN = '\033[97m'  # White
    GREY = '\033[90m'  # Light grey
    ENDC = '\033[0m'   # Reset color


# Load the CSV file into a DataFrame
file_path = 'checkpoint/cleaned.csv'
df = pd.read_csv(file_path)

# Initialize index and file to store index state
index_file = 'checkpoint/index.txt'
index = 588

# Check if the index file exists and load the index if it does
if os.path.exists(index_file):
    with open(index_file, 'r') as f:
        index = int(f.read().strip())

# Iterate through each row in the DataFrame starting from the saved index
while index < len(df):
    row = df.iloc[index]
    current_line_number = index + 1
    
    # Print context in blue and reply in green
    print(f"[{current_line_number}], {colors.GREY}Context:{colors.ENDC} {colors.BLUE}{row['context']}{colors.ENDC}, \n{colors.GREY}Reply:{colors.ENDC} {colors.GREEN}{row['reply']}{colors.ENDC}")
    
    user_input = input(f"{colors.GREY}Press 'q' and Enter to delete this line or just Enter to skip:{colors.ENDC} ")
    
    if user_input.lower() == 'q':
        # Drop the row and reset index
        df = df.drop(index).reset_index(drop=True)
        print(f"{colors.GREY}Line deleted.\n{colors.ENDC}")
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print("Changes saved to cleaned.csv")
        # Decrement index after deletion
        #index -= 1
    else:
        index += 1
        print(f"{colors.GREY}Line skipped.\n{colors.ENDC} ")
    
    # Save the current index state to the index file
    with open(index_file, 'w') as f:
        f.write(str(index))
