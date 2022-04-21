import numpy as np
import torch

def set_random_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)