import os
import subprocess

def check_cuda_version():
    try:
        # Run nvcc --version command to get CUDA version
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        lines = output.strip().split('\n')
        
        # Extract CUDA version from the output
        for line in lines:
            if 'release' in line.lower():
                # Split line and get the CUDA version
                parts = line.strip().split(',')
                if len(parts) > 1:
                    version_info = parts[1].strip().split()
                    if len(version_info) > 1:
                        cuda_version = version_info[1]
                        return cuda_version
        
        return None  # Return None if CUDA version not found

    except FileNotFoundError:
        return None  # Return None if nvcc command not found (CUDA not installed)

    except Exception as e:
        print(f"An error occurred while checking CUDA version: {e}")
        return None

def check_cuda_home():
    """
    Check if CUDA_HOME environment variable is set correctly.
    """
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home
    else:
        return None

try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False

def check_gpu():
    """
    Check if GPU is available and set as device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_pytorch_version():
    """
    Print PyTorch version.
    """
    print(f"PyTorch version: {torch.__version__}")

def print_cuda_info():
    """
    Print CUDA version and path if available.
    """
    if torch.cuda.is_available():
        print(f"Using CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
        print(f"CUDA path: {torch.utils.cpp_extension.CUDA_HOME}")
    else:
        print("CUDA is not available. Running on CPU.")

# Main function to execute the checks
def main():
    if torch_installed:
        print("Torch is currently installed.")
        device = check_gpu()
        print(f"Device set to: {device}")
        print_pytorch_version()
        print_cuda_info()
    else:
        print("Torch is not currently installed.")
    
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"Detected CUDA version: {cuda_version}")
        print(f"To attempt to use GPU, please uninstall the current PyTorch version by running: pip uninstall torch")
        latest_version = "x.x.x"
        print("Replace x.x.x with the latest or compatible version for your CUDA version.")
        print(f"Install it like this: pip install torch=={latest_version}+cu{cuda_version.replace('.', '')} -f https://download.pytorch.org/whl/torch_stable.html")
    else:
        print("CUDA is not installed or nvcc command not found.")
    
    cuda_home = check_cuda_home()
    if cuda_home:
        print(f"CUDA_HOME is set to: {cuda_home}")
    else:
        print("CUDA_HOME is not set correctly or does not exist.")

if __name__ == "__main__":
    main()
