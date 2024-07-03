import codecs

def check_for_bom(file_path):
    """
    Check if a file has a BOM (Byte Order Mark) at the beginning.

    Args:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if BOM is found, False otherwise.
    """
    with open(file_path, 'rb') as file:
        content = file.read(4)  # Read the first 4 bytes

    return content.startswith(codecs.BOM_UTF8)

# Example usage
file_path = 'cleaned.csv'
has_bom = check_for_bom(file_path)

if has_bom:
    print(f"The file '{file_path}' has a BOM.")
else:
    print(f"The file '{file_path}' does not have a BOM.")
