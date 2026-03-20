import pandas as pd
import os
import atexit
import tkinter as tk
from tkinter import filedialog

# ANSI escape codes for colors
class colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    GREY = '\033[90m'
    ENDC = '\033[0m'

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()

# Prompt user to select a CSV file
file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Initialize index from saved state (default to 0)
index_file = 'checkpoint/index.txt'
index = 0

if os.path.exists(index_file):
    with open(index_file, 'r') as f:
        content = f.read().strip()
        if content:
            index = int(content)

pending_deletes = []
SAVE_EVERY = 20


def flush_pending():
    """Flush any pending deletes to disk. Called on normal exit and Ctrl+C."""
    global df, pending_deletes, index
    if pending_deletes:
        df = df.drop(pending_deletes).reset_index(drop=True)
        index -= len(pending_deletes)
        pending_deletes = []
        df.to_csv(file_path, index=False)
        print(f"\n{colors.GREY}Pending deletes flushed and saved.{colors.ENDC}")
    with open(index_file, 'w') as f:
        f.write(str(index))


# Ensure pending deletes are saved even on Ctrl+C or unexpected exit
atexit.register(flush_pending)

while index < len(df):
    row = df.iloc[index]
    current_line_number = index + 1

    print(f"[{current_line_number}], {colors.GREY}Context:{colors.ENDC} {colors.BLUE}{row['context']}{colors.ENDC}, \n{colors.GREY}Reply:{colors.ENDC} {colors.GREEN}{row['reply']}{colors.ENDC}")

    user_input = input(f"{colors.GREY}Press 'q' and Enter to delete this line or just Enter to skip:{colors.ENDC} ")

    if user_input.lower() == 'q':
        pending_deletes.append(index)
        print(f"{colors.GREY}Line marked for deletion.\n{colors.ENDC}")
    else:
        print(f"{colors.GREY}Line skipped.\n{colors.ENDC}")

    index += 1

    # Batch save periodically
    if len(pending_deletes) >= SAVE_EVERY:
        df = df.drop(pending_deletes).reset_index(drop=True)
        index -= len(pending_deletes)
        pending_deletes = []
        df.to_csv(file_path, index=False)
        print(f"{colors.GREY}Changes saved.{colors.ENDC}")

    # Save index state
    with open(index_file, 'w') as f:
        f.write(str(index))

# Final save for any remaining deletes (also caught by atexit)
flush_pending()
