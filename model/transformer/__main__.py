
import torch
from . import *
from .encoder import TransformerEncoder

def main():
    encoder = TransformerEncoder()
    input = torch.rand(2, 768)
    print(input)




if __name__ == '__main__':
    main()