
from model import BertConfig, TransformerConfig

def ToEncoderConf(bert_config):
    """Convert Bert Config to Transformer Config"""
    transformer_config = TransformerConfig()
    transformer_config.hidden_size = bert_config.hidden_size
    transformer_config.num_heads = bert_config.num_attention_heads
    transformer_config.num_blocks = bert_config.num_hidden_layers
    transformer_config.dropout = bert_config.hidden_dropout_prob
    transformer_config.forward_inner_size = bert_config.intermediate_size

    return transformer_config