import numpy as np


class Activation(object):
    """
    An abstract base class for an activation function that computes both the
    original activation function (the forward pass) as well as its gradient
    (the backward pass).

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    These classes have no __init__().

    """

    def forward(self, X, w):
        """
        Computes the forward pass through the activation function.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            loss - (float) Weights .
        """
        pass

    def backward(self, X, w):
        """
        Computes the gradient of the activation function with respect to the model
        parameters.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***


        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass



class Sigmoid(Activation):
    """
    This class defines the sigmoid activation function.
    """

    def forward(self, X, w):
        """
        Computes the forward pass through the sigmoid activation function.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            loss - (float) Weights.
        """
        pass

    def backward(self, X, w):
        """
        Computes the gradient of the sigmoid activation function with respect
        to the model parameters.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class Softmax(Activation):
    """

    """

    def forward(self, X, w):
        """
        Computes the forward pass through the sigmoid activation function.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            loss - (float) Weights.
        """
        pass

    def backward(self, X, w):
        """
        Computes the gradient of the sigmoid activation function with respect
        to the model parameters.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass