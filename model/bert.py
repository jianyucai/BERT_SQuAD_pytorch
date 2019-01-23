
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import EmbeddingLayer
from .transformer import TransformerEncoder
from .config import BertConfig
from utils import ToEncoderConf
from torch.nn import CrossEntropyLoss


class BertBasic(nn.Module):
    r"""Basic Bert Model
    Bert Basic consists following components
        (1): Embedding Layer
        (2): Transformer Encoder

    Args:
        config: BertConfig

    Inputs:
        input_ids: Input tokens into token ids,
            dimension: [batch_size, sequence_len]

        token_type_ids: Token type, 0 for sentence A, 1 for sentence B
            dimension: [batch_size, sequence_len]

        attention_mask: Mask on the input sentence for the attention
            module to see the boundary of input sequence, 0 means
            outside input sentence, and 1 means inside input sentence,
            dimension: [batch_size, sequence_len]

    """
    def __init__(self, config):
        super(BertBasic, self).__init__()
        self.embeddings = EmbeddingLayer(config)
        self.encoder = TransformerEncoder(ToEncoderConf(config))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # expand attention_mask from [batch_size, sequence_len]
        # to [batch_size, 1, 1, sequence_len], and the broadcast
        # mechanism will be used when add mask with scores
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        embeddings = self.embeddings(input_ids, token_type_ids)
        output = self.encoder(embeddings, attention_mask)
        return output


class BertForSquad(nn.Module):
    r"""BERT Model For SQuAD
    Basic BERT + Output Layer

    The output layer outputs a start position and end position
    probability for each token in the input sequence.

    That is,
    [batch_size, sequence_len, hidden_size]
    => [batch_size, sequence_len, 2]

    Note that when training, the Softmax is contained
    in CrossEntropyLoss, in testing, the return values
    start and end scores have not been through the
    Softmax.

    Args:
        config: BertConfig

    Inputs:
        input_ids: Input tokens into token ids,
            dimension: [batch_size, sequence_len]

        token_type_ids: Token type, 0 for sentence A, 1 for sentence B
            dimension: [batch_size, sequence_len]

        attention_mask: Mask on the input sentence for the attention
            module to see the boundary of input sequence, 0 means
            outside input sentence, and 1 means inside input sentence,
            dimension: [batch_size, sequence_len]

        start_positions: Act as label, the start position of the answer
            based on input paragraph and question,
            dimension [batch_size]

        end_positions: Act as label, the end position of the answer
            based on input paragraph and question,
            dimension [batch_size]
    """

    def __init__(self, config):
        super(BertForSquad, self).__init__()
        self.bert = BertBasic(config)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        hidden_output = self.bert(input_ids, token_type_ids, attention_mask)
        scores = self.output_layer(hidden_output)

        # split [batch_size, sequence_len, 2] into 2 [batch_size, sequence_len, 1]
        start_scores, end_scores = torch.split(scores, 1, dim=-1)
        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)

        # training
        if start_positions is not None and end_positions is not None:
            ignored_index = start_scores.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_scores, start_positions)
            end_loss = loss_fct(end_scores, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss

        else:  # prediction
            return start_scores, end_scores





