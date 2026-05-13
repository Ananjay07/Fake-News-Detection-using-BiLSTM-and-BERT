import os
import sys

# Path to torch/lib
torch_lib_path = r'C:\Users\Ananjay Pampalli\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\lib'

if os.path.exists(torch_lib_path):
    print(f"Adding {torch_lib_path} to DLL search path...")
    os.add_dll_directory(torch_lib_path)

try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Failed to import torch:")
    import traceback
    traceback.print_exc()
