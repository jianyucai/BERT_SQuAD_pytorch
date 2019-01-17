import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import attention
from .utils import cloneModule, gelu
from .config import TransformerConfig

class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention
    args:
        config: TransformerConfig
    """

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size: %d is not divisible to num_heads: %d"
                             % (self.hidden_size, self.num_heads))

        self.d_k = int(self.hidden_size / self.num_heads) # d_k = d_v
        # query, key, value
        self.linear_layers = cloneModule(nn.Linear(self.hidden_size, self.hidden_size), 4)
        self.dropout_layer = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # prepare for h heads,
            # resulting dimension (1, 1, hidden_size, hidden_size)
            mask = mask.unsqueeze(1)

        batch_num = query.size(0)

        # 1. Linear Projection for query, key and value,
        #    resulting dimension (batch_num, h, batch_size, d_k)
        query = self.linear_layers[0](query) \
                .view(batch_num, -1, self.num_heads, self.d_k) \
                .transpose(1, 2)
        key = self.linear_layers[1](key) \
                .view(batch_num, -1, self.num_heads, self.d_k) \
                .transpose(1, 2)
        value = self.linear_layers[2](value) \
                .view(batch_num, -1, self.num_heads, self.d_k) \
                .transpose(1, 2)

        # 2. Apply Scaled Dot Attention
        attn_output, _ = attention(query, key, value, mask, None)

        # 3. Concatenate the num_heads headers
        # (batch_num, h, batch_size, d_k) => (batch_num, batch_size, h, d_k)
        # => (batch_num, batch_size, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous() \
                      .view(batch_num, -1, self.hidden_size)

        # 4. Apply the last Linear Projection
        output = self.linear_layers[3](attn_output)

        return output


class FeedForwardNetwork(nn.Module):
    r"""Fully Connected Feed-Forward Network
    FFN(x) = GELU(x W_1 + b_1)W_2 + b_2

    data dimension changes through layers:
        hidden_size => inner_size => hidden_size

    args:
        config: TransformerConfig
    """
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = config.hidden_size
        self.inner_size = config.forward_inner_size
        # parameters
        self.inner_layer = nn.Linear(self.hidden_size, self.inner_size)
        self.outer_layer = nn.Linear(self.inner_size, self.hidden_size)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.outer_layer(self.dropout_layer(
            gelu(self.inner_layer(x))))


class ResidualConnection(nn.Module):
    r"""Residual Connection: Add & Norm
    Output:
        LayerNorm(x + SubLayer(x))
    """

    def __init__(self, config):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias