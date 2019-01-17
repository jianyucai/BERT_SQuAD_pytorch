
import torch.nn as nn
from .layers import EmbeddingLayer
from .transformer import TransformerEncoder
from .config import BertConfig
from utils import ToEncoderConf



class BertBasic(nn.Module):
    r"""Basic Bert Model
    Bert Basic consists following components
        (1): Embedding Layer
        (2): Transformer Encoder
    """
    def __init__(self, config):
        super(BertBasic, self).__init__()
        self.embeddings = EmbeddingLayer(config)
        self.encoder = TransformerEncoder(ToEncoderConf(config))

    def forward(self, input_ids, token_type_ids):
        embeddings = self.embeddings(input_ids, token_type_ids)
        output = self.encoder(embeddings)
        return output