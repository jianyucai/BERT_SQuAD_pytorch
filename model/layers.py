
import torch
import torch.nn as nn
from .transformer import LayerNorm


class EmbeddingLayer(nn.Module):
    r"""Embedding Layer
    The Embedding Layer has 3 important sublayers:
    (1): Token Embeddings
    (2): Segment Embeddings
    (3): Position Embeddings

    and also
    (4): Layer Normalization
    (5): Dropout

    The input is a batch of token ids, with size (batch_size, token_num)

    args:
        BertConfig
    """

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        num_tokens = input_ids.size(1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arrange(num_tokens, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

