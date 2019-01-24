import tensorflow as tf
from model import BertConfig, BertBasic
import numpy as np
import torch
import torch.nn as nn
import argparse

"""
TensorFlow checkpoint has 3 files:
    (1) A meta file
    (2) A index file
    (3) A data file

Example:
    .
    ├── bert_model.ckpt.data-00000-of-00001
    ├── bert_model.ckpt.index
    ├── bert_model.ckpt.meta

To read parameter information in those files, you have to
use a TensorFlow API:
    tf.train.list_variables(checkpoint_file)

The parameter should be the prefix of those 3 files. And 
the return is a list of tuple [(parameter_name, size)]

Example:
    >>> checkpoint_file = "/Users/apple/Desktop/uncased_L-12_H-768_A-12/bert_model.ckpt"
    >>> init_vars = tf.train.list_variables(checkpoint_file)
    
After reading the names and sizes of parameters, we can now try
to load those matrices.

Example:
    >>> array = tf.train.load_variable(checkpoint_file_prefix, name)
    
Now the array is of type `ndarray`, we need to transform it into 
type `torch.Parameter`, so it is assignable to `weights` and `bias`
members of a module.

Our procedure is as follows:
    ndarray -> torch.FloatTensor -> torch.Parameter

Example:
    >>> param = nn.Parameter(torch.from_numpy(array))

Before we try to assign those parameters to our model,
we should be careful to the minor differences between
the names of parameters of PyTorch and TensorFlow.

 ______________________________________________
| PyTorch | TensorFlow                         |
|----------------------------------------------|
| weight  |  kernel / gamma / output_weights   |
| bias    |  beta / bias / output_bias         |
|----------------------------------------------|


"""

def parameter(name, param_dict, transpose=False):
    """Transform a numpy matrix to PyTorch Parameter"""
    if transpose is True:
        return nn.Parameter(torch.from_numpy(np.transpose(param_dict[name])))
    return nn.Parameter(torch.from_numpy(param_dict[name]))

def load_tf_checkpoint(checkpoint_file_prefix, config_file, pytorch_save_path):
    """
    Procedures:
    (1) Load the parameters in TensorFlow checkpoint files
    (2) Extract corresponding parameters and assign to a pytorch model
    (3) Store the parameters in pytorch-way
    """

    param_info = tf.train.list_variables(checkpoint_file_prefix)
    names = []
    arrays = []
    for name, shape in param_info:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(checkpoint_file_prefix, name)
        names.append(name)
        arrays.append(array)

    param_dict = dict((name, array) for (name, array) in zip(names, arrays))

    bert_conf = BertConfig.from_json_file(config_file)
    bert_model = BertBasic(bert_conf)

    """Embedding Layers"""
    # bert/embeddings/word_embeddings
    name = "bert/embeddings/word_embeddings"
    bert_model.embeddings.word_embeddings.weight = parameter(name, param_dict)

    # bert/embeddings/position_embeddings
    name = "bert/embeddings/word_embeddings"
    bert_model.embeddings.position_embeddings.weight = parameter(name, param_dict)

    # bert/embeddings/token_type_embeddings
    name = "bert/embeddings/token_type_embeddings"
    bert_model.embeddings.token_type_embeddings.weight = parameter(name, param_dict)

    # bert/embeddings/LayerNorm/beta
    name = "bert/embeddings/LayerNorm/beta"
    bert_model.embeddings.norm.bias = parameter(name, param_dict)

    # bert/embeddings/LayerNorm/gamma
    name = "bert/embeddings/LayerNorm/gamma"
    bert_model.embeddings.norm.weight = parameter(name, param_dict)

    """Encoder Layers"""
    for layer_idx in range(bert_conf.num_hidden_layers):
        prefix = "bert/encoder/layer_" + str(layer_idx) + "/"

        # bert/encoder/layer_0/attention/output/LayerNorm/beta
        name = prefix + "attention/output/LayerNorm/beta"
        bert_model.encoder.block_list[layer_idx].residual_layer_1.bias = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/output/LayerNorm/gamma
        name = prefix + "attention/output/LayerNorm/gamma"
        bert_model.encoder.block_list[layer_idx].residual_layer_1.weight = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/output/dense/bias
        name = prefix + "attention/output/dense/bias"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[3].bias = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/output/dense/kernel
        name = prefix + "attention/output/dense/kernel"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[3].weight = \
            parameter(name, param_dict, transpose=True)

        # bert/encoder/layer_0/attention/self/query/bias
        name = prefix + "attention/self/query/bias"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[0].bias = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/self/query/kernel
        name = prefix + "attention/self/query/kernel"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[0].weight = \
            parameter(name, param_dict, transpose=True)

        # bert/encoder/layer_0/attention/self/key/bias
        name = prefix + "attention/self/key/bias"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[1].bias = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/self/key/kernel
        name = prefix + "attention/self/key/kernel"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[1].weight = \
            parameter(name, param_dict, transpose=True)

        # bert/encoder/layer_0/attention/self/value/bias
        name = prefix + "attention/self/value/bias"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[2].bias = parameter(name, param_dict)

        # bert/encoder/layer_0/attention/self/value/kernel
        name = prefix + "attention/self/value/kernel"
        bert_model.encoder.block_list[layer_idx].attention_layer.linear_layers[2].weight = \
            parameter(name, param_dict, transpose=True)

        # bert/encoder/layer_0/intermediate/dense/bias
        name = prefix + "intermediate/dense/bias"
        bert_model.encoder.block_list[layer_idx].feed_forward_layer.inner_layer.bias = parameter(name, param_dict)

        # bert/encoder/layer_0/intermediate/dense/kernel
        name = prefix + "intermediate/dense/kernel"
        bert_model.encoder.block_list[layer_idx].feed_forward_layer.inner_layer.weight = \
            parameter(name, param_dict, transpose=True)

        # bert/encoder/layer_0/output/LayerNorm/beta
        name = prefix + "output/LayerNorm/beta"
        bert_model.encoder.block_list[layer_idx].residual_layer_2.bias = parameter(name, param_dict)

        # bert/encoder/layer_0/output/LayerNorm/gamma
        name = prefix + "output/LayerNorm/gamma"
        bert_model.encoder.block_list[layer_idx].residual_layer_2.weight = parameter(name, param_dict)

        # bert/encoder/layer_0/output/dense/bias
        name = prefix + "output/dense/bias"
        bert_model.encoder.block_list[layer_idx].feed_forward_layer.outer_layer.bias = parameter(name, param_dict)

        # bert/encoder/layer_0/output/dense/kernel
        name = prefix + "output/dense/kernel"
        bert_model.encoder.block_list[layer_idx].feed_forward_layer.outer_layer.weight = \
            parameter(name, param_dict, transpose=True)

    print("Save PyTorch model parameters in {}".format(pytorch_save_path))
    torch.save(bert_model.state_dict(), pytorch_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_file_prefix",
                        default=None,
                        type=str,
                        required=True,
                        help="Prefix of TensorFlow checkpoint files")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="config_file of TensorFlow checkpoint files")

    parser.add_argument("--pytorch_save_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Save path of PyTorch model parameters")

    args = parser.parse_args()

    load_tf_checkpoint(args.checkpoint_file_prefix,
                       args.config_file,
                       args.pytorch_save_path)