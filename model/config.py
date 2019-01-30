
import json


class BertConfig(object):
    r"""Configuration for BERT
    In order to read configs from Google pre-trained models,
    we choose to be consistent with the configs provided by
    Google.

    args:
        attention_probs_dropout_prob:
            dropout rate for attention, default: 0.1
        hidden_act:
            activation function, default: gelu
        hidden_dropout_prob:
            dropout rate for fully connected layers, default: 0.1
        hidden_size:
            hidden state size, default: 768
        initializer_range:
            The sttdev of the truncated_normal_initializer for
            initializing all weight matrices, default: 0.2
        intermediate_size:
            The size for intermediate layer in Feed Forward Layer,
            default: 3072
        max_position_embeddings:
            Maximum input sequence length, default: 512
        num_attention_heads:
            Number of heads in attention, default: 12
        num_hidden_layers:
            Number of Encoder blocks, default: 12
        type_vocab_size:
            The vocabulary size of the `token_type_ids`, default: 2
        vocab_size:
            Vocabulary size for input sequences, default: 30522 (by WordPiece)
    """

    def __init__(self,
                 attention_probs_dropout_prob=0.1,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 initializer_range=0.02,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 num_attention_heads=12,
                 num_hidden_layers=12,
                 type_vocab_size=2,
                 vocab_size=30522):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size

    @classmethod
    def from_dict(cls, json_object):
        """Read parameters from dictionary"""
        config = BertConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Read parameters from json file"""
        with open(json_file, "r", encoding='utf-8') as file:
            text = file.read()
        return cls.from_dict(json.loads(text))