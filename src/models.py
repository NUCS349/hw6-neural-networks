import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError()


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()

    def forward(self, inputs):
r       raise NotImplementedError()
