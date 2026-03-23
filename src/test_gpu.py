import torch

print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print("SUCCESS: Your RTX 3050 is linked and ready!")
else:
    print("FAILURE: GPU not detected. Using CPU.")