
import torch.nn as nn
from .sublayers import MultiHeadAttention, FeedForwardNetwork, ResidualConnection
from .utils import cloneModule


class TransformerEncoderBlock(nn.Module):
    r"""Transformer Encoder Block
    The unit of Transformer Encoder,
    a encoder may consists of a stack of blocks.

    A Block consists (in order):
        (1) A Self-Attention Layer
        (2) A Residual Layer [1]
        (3) A Feed Forward Layer
        (4) A Residual Layer [2]

    args:
        config: TransformerConfig
    """

    def __init__(self, config):
        super(TransformerEncoderBlock, self).__init__()
        self.attention_layer = MultiHeadAttention(config)
        self.residual_layer_1 = ResidualConnection(config)
        self.feed_forward_layer = FeedForwardNetwork(config)
        self.residual_layer_2 = ResidualConnection(config)

    def forward(self, input, attention_mask):
        input = self.residual_layer_1(input, lambda x: self.attention_layer(x, x, x, attention_mask))
        return self.residual_layer_2(input, self.feed_forward_layer)


class TransformerEncoder(nn.Module):
    r"""Transformer Encoder
    A stack of Transformer Encoder Blocks

    args:
        config: TransformerConfig
    """

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.block_list = cloneModule(TransformerEncoderBlock(config), config.num_blocks)

    def forward(self, input, attention_mask):
        for block in self.block_list:
            input = block(input, attention_mask)
        return input