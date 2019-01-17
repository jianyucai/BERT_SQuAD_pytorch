

class TransformerConfig:
    """Transformer Config
    args:
        hidden_size:
            embedding size, or input size of encode layer
        num_heads:
            number of attention heads
        num_blocks:
            number of encoder blocks
        dropout:
            dropout rate
        forward_inner_size:
            inner layer size of feed forward layer

    """
    def __init__(self, hidden_size=768, num_heads=12, num_blocks=12, dropout=0.1, forward_inner_size=3072):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.forward_inner_size = forward_inner_size
