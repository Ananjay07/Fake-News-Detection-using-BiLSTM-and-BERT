import os
import sys
import ctypes

def check_dll(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    print(f"Checking {path}...")
    try:
        # Try to load it manually
        ctypes.WinDLL(path)
        print(f"Successfully loaded {path}")
    except Exception as e:
        print(f"Failed to load {path}: {e}")

torch_lib = r"C:\Users\Ananjay Pampalli\anaconda3\Lib\site-packages\torch\lib"
print(f"Torch lib path: {torch_lib}")

if os.path.exists(torch_lib):
    print("Adding torch lib to DLL directory...")
    os.add_dll_directory(torch_lib)

# Check c10.dll specifically
c10_path = os.path.join(torch_lib, "c10.dll")
check_dll(c10_path)

try:
    import torch
    print("Torch imported successfully!")
    print(f"Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print("\nImport failed:")
    import traceback
    traceback.print_exc()
