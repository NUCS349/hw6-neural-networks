import numpy as np

from .activations import Sigmoid, Softmax


class DenseLayer(object):
    """

    """

    def __init__(self, activation, num_nodes, weights=None):

        if activation == "sigmoid":
            self.activation = Sigmoid
        elif activation == "softmax":
            self.activation = Softmax
        else:
            raise ValueError(f'Unknown activation type {activation}')


    def forward(self, ):
        """

        :return:
        """

    def backward(self, ):
        """

        :return:
        """