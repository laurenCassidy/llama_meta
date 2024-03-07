import torch
print("Torch Version: ", torch.__version__)
print("CUDA is available: ",torch.cuda.is_available())
print("CUDA toolkit version: ", torch.version.cuda)
if torch.cuda.is_available():
    print("Number of GPUs: ", torch.cuda.device_count())
    print("Name of first GPU: ",torch.cuda.get_device_name(0))
    print("Current CUDA device being used: ",torch.cuda.current_device())

