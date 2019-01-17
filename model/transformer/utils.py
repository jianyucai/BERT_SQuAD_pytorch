
import math
import torch
import torch.nn as nn
import copy

def gelu(x):
    r"""GELU activation function
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def cloneModule(module, N):
    """Clone N identical modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
