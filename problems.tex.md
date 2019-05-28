# Coding (6 points)

For the coding part, you will have to implement a fully connected (aka dense) neural network using only `numpy`. Your neural network will have to use the backpropagate algorithm to update the model's weights for each iteration. You will write the following code:

- Two types of activations functions (in `your_code/activations.py`)
- One loss function, cross entropy (in `your_code/loss.py`)
- A fully connected layer class (in `your_code/fc_layer.py`)
- A neural network that puts everything together (in `your_code/network.py`)

It is recommended that you write your code in the above order.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you may move on to the next part - using PyTorch.

Your grade for this section is defined by the autograder. If it says you got an 80/100, you get 4 points here. 


# Free-response questions (4 points)


To answer some of the free-response questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment.


## PyTorch Introduction

For this assigment, you will be learning about a popular library for implementing neural networks called PyTorch. It is very popular to train neural networks using a GPU (because they speed up training by a large factor), but for this assignment not need a GPU as all of the training will happen on your CPU. There are other popular libraries for neural networks, such as TensorFlow, but for this assignment you will be using PyTorch.

Here is the official website for PyTorch: [https://pytorch.org/](https://pytorch.org/)

Here is the API documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

Here is a cheat sheet of commonly used methods: [https://pytorch.org/tutorials/beginner/ptcheat.html](https://pytorch.org/tutorials/beginner/ptcheat.html)

Here is a comparison of PyTorch and Numpy methods: [https://github.com/wkentaro/pytorch-for-numpy-users](https://github.com/wkentaro/pytorch-for-numpy-users)


## Understanding PyTorch (1 point)

Read the first four tutorials on [this page](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) ("*What is PyTorch?*", "*Autograd: Automatic Differentiation*", "*Neural Networks*", and "*Training a Classifier*") and then answer the following questions.

1. What is a tensor? How is it different than a matrix? (0.25 points)

2. What is automatic differentiation? How does it relate to gradient descent and back-propagation? Why is it important for PyTorch? (0.25 points)

3. Why does PyTorch have its own tensor class (`torch.Tensor`) when it is extremely similar to numpy's `np.ndarray`? (0.25 points)

4. 


## Training a network on CIFAR (1 point)

Read the website for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), but you **do not need to download the dataset through your browser**. You can use PyTorch to download and load the dataset for you. 

5. How many images are in CIFAR-10? What kinds of images are in this dataset? How many classes are there? What is the format of the images? (0.25 points)

6. Show a graph of three different, random images from the same class. Do you think this task will be easier or hard than training on MNIST? Why? What are the baseline error rates for this dataset? (0.25 points)

7. Using PyTorch, download and train a neural network to classify images from the CIFAR-10 dataset. Use a network with three hidden layers of size 

## Hyperparameter search (1 point)

## Convolutional Layers and Dropout (1 point)