"""
Diagnostic tool for SimpleGPT training environment.
Scans project files for dependencies, checks installs,
checks PyTorch/CUDA/GPU, and offers to auto-install missing packages.
"""

import ast
import glob
import os
import sys
import subprocess
import shutil
import re

# Packages that are always required regardless of imports
ALWAYS_REQUIRED = {"torch", "pandas"}

# Map import names to pip package names (where they differ)
IMPORT_TO_PIP = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "pandas": "pandas",
    "requests": "requests",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "tkinter": None,  # stdlib, can't pip install
}

# Standard library modules to ignore
STDLIB = {
    "os", "sys", "time", "re", "json", "math", "random", "collections",
    "typing", "datetime", "subprocess", "shutil", "hashlib", "atexit",
    "functools", "itertools", "pathlib", "glob", "io", "copy", "abc",
    "dataclasses", "enum", "string", "textwrap", "struct", "pickle",
    "csv", "argparse", "logging", "unittest", "tkinter", "ast",
}


def section(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def scan_project_imports():
    """Scan all .py files in the project for third-party imports."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    py_files = glob.glob(os.path.join(project_dir, "*.py"))

    imports = set()
    for filepath in py_files:
        # Skip this file
        if os.path.basename(filepath) == "torch-debug.py":
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except Exception:
            pass

    # Add always-required packages
    imports.update(ALWAYS_REQUIRED)

    # Filter out stdlib and local project files
    local_files = {os.path.splitext(os.path.basename(f))[0] for f in py_files}
    third_party = set()
    for imp in imports:
        if imp in STDLIB:
            continue
        if imp in local_files:
            continue
        third_party.add(imp)

    return third_party


def get_pip_name(import_name):
    """Convert an import name to a pip package name."""
    if import_name in IMPORT_TO_PIP:
        return IMPORT_TO_PIP[import_name]
    return import_name


def get_vram(props):
    """Get total VRAM from device properties, compatible across PyTorch versions."""
    return getattr(props, 'total_memory', getattr(props, 'total_mem', 0))


def detect_nvidia_gpu():
    """Check if an NVIDIA GPU exists using nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
    except Exception:
        pass
    return None


def detect_system_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=5)
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass

    return None


def get_recommended_install(cuda_version_str):
    """Returns the recommended pip install command based on system CUDA version."""
    if cuda_version_str is None:
        return None, None

    try:
        cuda_ver = float(cuda_version_str)
    except ValueError:
        return None, None

    if cuda_ver >= 13.0:
        tag = "cu130"
        label = "CUDA 13.0"
    elif cuda_ver >= 12.8:
        tag = "cu128"
        label = "CUDA 12.8"
    elif cuda_ver >= 12.6:
        tag = "cu126"
        label = "CUDA 12.6"
    elif cuda_ver >= 12.0:
        tag = "cu126"
        label = "CUDA 12.6 (compatible with your CUDA 12.x)"
    elif cuda_ver >= 11.8:
        tag = "cu126"
        label = "CUDA 12.6 (update your CUDA toolkit for best results)"
    else:
        return None, f"Your CUDA {cuda_version_str} is very old. Update your NVIDIA drivers first."

    cmd = f"pip3 install torch torchvision --index-url https://download.pytorch.org/whl/{tag}"
    return cmd, label


def check_python():
    section("Python")
    print(f"  Version:    {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print(f"  Platform:   {sys.platform}")


def check_torch():
    section("PyTorch")
    try:
        import torch
    except ImportError:
        print("  [FAIL] PyTorch is NOT installed.")
        print("  Fix:   pip install torch")
        print("         https://pytorch.org/get-started/locally/")
        return False

    print(f"  Version:    {torch.__version__}")
    print(f"  Build:      {'CUDA' if torch.version.cuda else 'CPU-only'}")
    if torch.version.cuda:
        print(f"  Built with: CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    return True


def check_nvidia():
    section("NVIDIA GPU")
    gpus = detect_nvidia_gpu()
    if gpus is None:
        print("  [FAIL] No NVIDIA GPU detected.")
        print()
        print("  SimpleGPT requires an NVIDIA GPU for training.")
        print("  AMD and Intel GPUs are not supported by PyTorch CUDA.")
        print()
        print("  If you DO have an NVIDIA GPU:")
        print("    - Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("    - Make sure nvidia-smi is on your PATH")
        return False
    else:
        for i, gpu in enumerate(gpus):
            print(f"  [OK] GPU {i}: {gpu}")
        return True


def check_cuda():
    section("CUDA")
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch not installed.")
        return

    system_cuda = detect_system_cuda_version()
    if system_cuda:
        print(f"  System CUDA version: {system_cuda}")
    else:
        print(f"  System CUDA version: not detected")

    if torch.cuda.is_available():
        print(f"  PyTorch CUDA:        {torch.version.cuda}")
        print(f"  [OK] CUDA is available and working.")
        print(f"  Device count:        {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = get_vram(props) / (1024 ** 3)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    VRAM:          {vram_gb:.1f} GB")
            print(f"    Compute:       {props.major}.{props.minor}")
            print(f"    SM count:      {props.multi_processor_count}")

            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            free = vram_gb - reserved
            print(f"    Allocated:     {allocated:.2f} GB")
            print(f"    Reserved:      {reserved:.2f} GB")
            print(f"    Free (approx): {free:.1f} GB")
    else:
        print(f"  [FAIL] CUDA is NOT available to PyTorch.")
        print()
        if not torch.version.cuda:
            print("  Reason: You installed the CPU-only build of PyTorch.")
        else:
            print(f"  Reason: PyTorch has CUDA {torch.version.cuda} but can't reach the GPU.")
            print("  Check:  - Are NVIDIA drivers up to date?")
            print("          - Does nvidia-smi work?")


def recommend_install():
    section("Recommended PyTorch Install")

    gpus = detect_nvidia_gpu()
    if gpus is None:
        print("  No NVIDIA GPU found. Cannot recommend a CUDA install.")
        print("  You can only use the CPU build:")
        print()
        print("    pip3 install torch torchvision")
        print()
        print("  WARNING: Training on CPU will be extremely slow.")
        return

    system_cuda = detect_system_cuda_version()
    if system_cuda is None:
        print("  NVIDIA GPU found but no CUDA toolkit detected.")
        print()
        print("  Install NVIDIA CUDA toolkit first:")
        print("    https://developer.nvidia.com/cuda-downloads")
        print()
        print("  Or if your drivers are recent enough, try:")
        cmd, _ = get_recommended_install("12.6")
        print(f"    {cmd}")
        return

    cmd, label = get_recommended_install(system_cuda)
    if cmd is None:
        print(f"  {label}")
        return

    print(f"  Your system CUDA: {system_cuda}")
    print(f"  Recommended:      {label}")
    print()
    print(f"  Run this command:")
    print(f"    {cmd}")
    print()

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  (You already have a working CUDA PyTorch — no action needed)")
        elif torch.version.cuda:
            print(f"  (Your current PyTorch has CUDA {torch.version.cuda} but it's not working — try reinstalling)")
        else:
            print(f"  (Your current PyTorch is CPU-only — run the command above to fix)")
    except ImportError:
        print(f"  (PyTorch not installed — run the command above)")


def check_dependencies():
    section("Dependencies (auto-scanned from project)")

    required = scan_project_imports()
    installed = {}
    missing = []

    for import_name in sorted(required):
        pip_name = get_pip_name(import_name)
        if pip_name is None:
            # stdlib like tkinter — skip
            continue
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            installed[import_name] = version
            print(f"  [OK]   {import_name:15s} {version}")
        except ImportError:
            print(f"  [FAIL] {import_name:15s} not installed  (pip: {pip_name})")
            missing.append((import_name, pip_name))

    if missing:
        print()
        pip_names = " ".join(pip for _, pip in missing)
        print(f"  Missing packages detected.")
        print(f"  Install command: pip install {pip_names}")
        print()
        answer = input("  Install missing packages now? [y/N]: ").strip().lower()
        if answer == "y":
            # Don't auto-install torch — it needs the CUDA index URL
            torch_missing = any(imp == "torch" for imp, _ in missing)
            non_torch = [(imp, pip) for imp, pip in missing if imp != "torch"]

            if non_torch:
                pip_cmd = [sys.executable, "-m", "pip", "install"] + [pip for _, pip in non_torch]
                print(f"\n  Running: {' '.join(pip_cmd)}")
                subprocess.run(pip_cmd)

            if torch_missing:
                print()
                print("  torch needs special install for CUDA support.")
                print("  See the 'Recommended PyTorch Install' section above.")
        else:
            print("  Skipped.")
    else:
        print()
        print(f"  All {len(installed)} dependencies installed.")


def check_training_readiness():
    section("Training Readiness")
    try:
        import torch
    except ImportError:
        print("  [FAIL] Can't check — PyTorch not installed.")
        return

    issues = []

    if not torch.cuda.is_available():
        issues.append("No GPU available — training will be extremely slow on CPU")
    else:
        vram = get_vram(torch.cuda.get_device_properties(0)) / (1024 ** 3)
        if vram < 8:
            issues.append(f"GPU has {vram:.1f} GB VRAM — may be tight for Phi-2 + LoRA (recommend 10+ GB)")

        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7:
            issues.append(f"GPU compute capability {capability[0]}.{capability[1]} — FP16 training may not work well (need 7.0+)")

    try:
        import peft
    except ImportError:
        issues.append("peft not installed — needed for LoRA fine-tuning")

    try:
        import transformers
    except ImportError:
        issues.append("transformers not installed — needed for model loading")

    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] Everything looks good for training.")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = get_vram(torch.cuda.get_device_properties(0)) / (1024 ** 3)
            print(f"  Ready to train on {name} ({vram:.1f} GB VRAM)")


def quick_gpu_test():
    section("Quick GPU Test")
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch not installed.")
        return

    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available.")
        return

    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("  [OK] GPU compute works (matmul test passed)")

        x = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("  [OK] FP16 compute works")

    except Exception as e:
        print(f"  [FAIL] GPU test failed: {e}")


if __name__ == "__main__":
    print("SimpleGPT Environment Diagnostic")
    check_python()
    torch_ok = check_torch()
    check_nvidia()
    check_cuda()
    recommend_install()
    check_dependencies()
    check_training_readiness()
    if torch_ok:
        quick_gpu_test()
    print(f"\n{'=' * 50}")
    print("  Done.")
    print(f"{'=' * 50}")
