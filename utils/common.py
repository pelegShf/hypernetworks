import random, torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)  # For the random lib
    torch.manual_seed(seed) # For CPU
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) # For GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
