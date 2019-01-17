
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout_layer=None):
    r"""Scaled Dot Product Attention
    Attention Mechanism can be expressed with a
    Query-Key-Value abstraction.

    The return value of Attention Mechanism is
    the weighted combination of the values, where
    the weights are calculated with keys and
    queries.

    args:
        query: size (batch_num, query_num, d_q)
        key:   size (batch_num, key_num, d_k)
        value: size (batch_num, value_num, d_v)
        mask:  type torch.ByteTensor

    returns:
        attention values: weighted sum of values
        scores:           weights
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # set value to -inf where mask = 1
        scores = scores.masked_fill_(mask, -np.inf)

    scores = F.softmax(scores, dim=-1)

    if dropout_layer is not None:
        # it seems unusual to dropout weights
        # but that's what Transformer paper do
        scores = dropout_layer(scores)

    return torch.matmul(scores, value), scores
