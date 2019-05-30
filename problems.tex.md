# Coding (0 points...but free response answers without supporting code may get a 0)

There is only one autograded test for this assignment and that is the test_netid.py test. You will not have to write any other code to pass the tests. You will, however, still have coding to do for the experiments.

You must hand in whatever code you did for data loading, your visualizations, and experiments by pushing to github (as you did for all previous assignments). Your code should be in the code/ directory.

**NOTE: if we have any doubts about your experiments we reserve the right to check this code to see if your results could have been generated using this code. If we don't believe it, or if there is no code at all, then you may receive a 0 for any free-response answer that would have depended on running code.**

You should make a conda environment for this homework just like you did for previous homeworks. We have included a requirements.txt.

# Free-response questions (10 points)


To answer some of the free-response questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment.


## PyTorch Introduction

For this assigment, you will be learning about a popular library for implementing neural networks called PyTorch. It is very popular to train neural networks using a GPU (because they speed up training by a large factor), but for this assignment **you do not need a GPU** as **all of the training will happen on your CPU**. There are other popular libraries for neural networks, such as TensorFlow, but for this assignment you will be using PyTorch.

Here is the official website for PyTorch: [https://pytorch.org/](https://pytorch.org/)

Here is the API documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

Here is a cheat sheet of commonly used methods: [https://pytorch.org/tutorials/beginner/ptcheat.html](https://pytorch.org/tutorials/beginner/ptcheat.html)

Here is a comparison of PyTorch and Numpy methods: [https://github.com/wkentaro/pytorch-for-numpy-users](https://github.com/wkentaro/pytorch-for-numpy-users)


## Understanding PyTorch (1 point)

Read the first four tutorials on [this page](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) ("*What is PyTorch?*", "*Autograd: Automatic Differentiation*", "*Neural Networks*", and "*Training a Classifier*") and then answer the following questions.

1. (0.25 points) What is a tensor? How is it different than a matrix?

2. (0.5 points) What is automatic differentiation? How does it relate to gradient descent and back-propagation? Why is it important for PyTorch?

3. (0.25 points) Why does PyTorch have its own tensor class (`torch.Tensor`) when it is extremely similar to numpy's `np.ndarray`?


## Training on MNIST (3 points)

4. (0.5 points) In a previous homework, you have had to use strategies such as One-Vs-All (OVA) or One-Vs-One (OVO) to train multiclass classifiers. Is this strategy something that you must implement for neural networks? Why or why not? (0.5 points)

For this next few questions, it may be helpful to know how to save and load trained models to and from disc. See [this tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) about how to do that. We recommend you read all of these questions prior to training your models in case you decide that you want to save your trained models to use for later questions.

The goal of the next three questions is to train neural networks to classify handwritten digits from the MNIST dataset and analyze the accuracy and training time of these neural networks.

5. (1 point) Select a subset of the MNIST training set with 100 examples and select a subset of the MNIST testing set with 100 examples (recall that you should be selecting examples in such a way that you minimize bias, i.e., make sure all ten digits are in your training and testing set). Train a neural network with two fully connected (aka, dense) hidden layers of size 128, and 64, respectively. Your network should have a total of four layers: an input layer that takes in examples, two hidden layers, and an output layer that outputs a predicted class. Your first three layers should have a ReLU activation function and your last (output) layer should have a softmax activation function. Train your model on your subset for 100 epochs with batch size 10. Make a graph of your training loss at every epoch. Report the time it took to run your network and the accuracy of your network on your training set. **Hint: In PyTorch the fully connected layers are called `torch.nn.Linear()`.**

6. (1 point) Using the same testing subset and network architecture as before, select a subset of the MNIST training set with 250, 500, 1000. Train a new model for each MNIST subset. Including your answers from question 5, make two graphs; one that shows the amount of training time along the y-axis and number of training examples along the x-axis, and a second that shows accuracy on your testing set on the y-axis and number of training examples on along the x-axis.

7. (0.5 points) What happens to your training time as the number of training examples increases? Roughly how many hours would you expect it to take to train on the full MNIST training set using the same architecture (on your CPU)? What happens to the accuracy as the number of training examples increases?


## DogSet (2 points)

We have provided you with a dataset called DogSet. DogSet is a subset from a popular machine learning dataset called ImageNet (more info [here](http://www.image-net.org/) and [here](https://en.wikipedia.org/wiki/ImageNet)). [The DogSet dataset is available here.](https://drive.google.com/open?id=1wlZZ8MBbcugcmiPqB4QJ8Jh9BdVFJSoC) (Note: you need to be signed into your `@u.northwestern.edu` google account to view this link). As it name implies, the entire dataset is comprised of images of dogs and labels indicating what dog breed is in the image. The metadata, which correlates any particular image with its label and training/testing partition, is provided in a file called `dogs.csv`. For this task, the testing set is called `valid` under the `partition` header of the provided csv file. We have provided a general data loader for you (in `data/dogs.py`), but you may need to adopt it to your needs when using PyTorch (there are many ways to do this, see the *Training a Classifier* tutorial above).

The goal of the next three questions is to analyze DogSet and train neural networks to classify dog breeds.

8. (0.5 points) How many images are in DogSet total? How many have labels? How many are in the `train` partition? How many in the `valid` partition? What are the dimensions of each image? What is the color palette of the images (greyscale, black & white, RBG)? How many dog breeds are there? (0.5 points)

9. (0.5 points) Select one type of breed. Look through variants of images of this dog breed. Show 3 different images of the same breed that you think are particularly challenging for a classifier to get correct. Explain why you think these three images might be challenging for a classifier. 

10. (0.5 point) Based on your experience with neural networks from the MNIST dataset, roughly determine a subset of the DogSet `train` and `valid` partition to train a neural network with the same architecture that you used for your MNIST networks in questions 5-7. Report the size of your training and testing sets, the number of epochs and batch size that you used. Also report the accuracy of your model and how long it took to train.

11. (0.5 points) Now, using the same `train` and `valid` DogSet subsets as in question 10, train a new network with _three_ fully connected hidden layers of size 256, 128, and 64 respectively (five layers total). As before, the first four layers should all use a ReLU activation function and the last layer should use softmax. Use the same number of epochs and batch size as in question 10. Report the accuracy of your model and how long it took to train. How did the accuracy change? How did the training time change? 


## Convolutional Layers and Pooling (4 points)

Convolutional layers are layers that sweep over and subsample their input in order to represent complex structures in the input layers. For more information about how they work, [see this blog post](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/). Don't forget to read the PyTorch documentation about Convolutional Layers (linked above).

12. Convolutional layers produce outputs that are of different size than their input by representing more than one input pixel with each node. If a 2D convolutional layer has `3` channels, batch size `16`, input size `(32, 32)`, padding `(4, 8)`, dilation `(1, 1)`, kernel size `(8, 4)`, and stride `(2, 2)`, what is the output size of the channel? (0.5 points)

12.  Combining convolutional layers with fully connected layers can provide a boon to scenarios involving learning from images.
13. Pooling question (0.5 points)



