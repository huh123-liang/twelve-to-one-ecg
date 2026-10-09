"""Print the environment so you know training will use the GPU."""
import platform
import sys

import torch

print("python :", sys.version.split()[0], platform.system())
print("torch  :", torch.__version__)
print("cuda   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu    :", torch.cuda.get_device_name(0))
    print("memory :", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
else:
    print("!! CUDA not available - reinstall PyTorch with the CUDA build (see README step 1)")
for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "wfdb", "tqdm", "tensorboard", "torchinfo", "tabulate"]:
    try:
        m = __import__(pkg)
        print(f"{pkg:7s}:", getattr(m, "__version__", "ok"))
    except ImportError:
        print(f"{pkg:7s}: MISSING -> pip install -r requirements.txt")
