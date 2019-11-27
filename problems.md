# Coding (5 points)

Your task is to implement neural networks and train them to perform classification on different types of data. You will write the following code:

- A function for training a model on a training dataset for one epoch (in `src/run_model.py`)
- A function for evaluating a trained model on a validation/testing dataset (in `src/run_model.py`)
- A function which trains (over multiple epoch), validates, or tests a model as specified (in `src/run_model.py`)
- Two fully connected and two convolutional neural networks (in `src/models.py`)

You will also have to write code for experiments. Note that the free response questions provide necessary information for writing the code above. Therefore, we recommend you write the code after going through the free response questions and follow the same order in running the experiments. 

Your grade for this section is defined by the autograder. If it says you got an 80/100, you get 4 points here.

You should make a conda environment for this homework just like you did for previous homeworks. We have included a requirements.txt.

# Free-response questions (9 points)

To answer the free-response questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment. You must hand in whatever code you write for experiments by pushing to github (as you did for all previous assignments). 

**NOTE: if we have any doubts about your experiments we reserve the right to check this code to see if your results could have been generated using this code. If we don't believe it, or if there is no code at all, then you may receive a 0 for any free-response answer that would have depended on running code.**


## PyTorch introduction

For this assignment, you will be learning about a popular library for implementing neural networks called PyTorch. It is very popular to train neural networks using a GPU (because they speed up training by a large factor), but for this assignment **you do not need a GPU** as **all of the training will happen on your CPU**. There are other popular libraries for neural networks, such as TensorFlow, but for this assignment you will be using PyTorch.

Here is the official website for PyTorch: [https://pytorch.org/](https://pytorch.org/)

Here is the API documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

Here is a cheat sheet of commonly used methods: [https://pytorch.org/tutorials/beginner/ptcheat.html](https://pytorch.org/tutorials/beginner/ptcheat.html)

Here is a comparison of PyTorch and Numpy methods: [https://github.com/wkentaro/pytorch-for-numpy-users](https://github.com/wkentaro/pytorch-for-numpy-users)

**IMPORTANT: PyTorch is not included in `requirements.txt`!** To install PyTorch, find the correct install command for your operating system and version of python [here](https://pytorch.org/get-started/locally/). For "PyTorch Build" select the `Stable (1.3)` build, select your operating system, for "Package" select `pip` , for "Language" select your python version (either python 3.5, 3.6, or 3.7), and finally for "CUDA" select `None`. **Make sure to run the command with your conda environment activated.**

_Note: To determine which python you are using, type `python --version` into your command line._


## Understanding PyTorch 

Read the first four tutorials on [this page](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) ("*What is PyTorch?*", "*Autograd: Automatic Differentiation*", "*Neural Networks*", and "*Training a Classifier*"). 

## Training on MNIST (1 point)

Let us train neural networks to classify handwritten digits from the MNIST dataset and analyze the accuracy and training time of these neural networks. 

**Build the model architecture:** Create a neural network with two fully connected (aka, dense) hidden layers of size 128, and 64, respectively. Your network should have a total of four layers: an input layer that takes in examples, two hidden layers, and an output layer that outputs a predicted class (10 possible classes, one for each digit class in MNIST). Your hidden layers should have a ReLU activation function.  Your last (output) layer should be a linear layer with one node per class (in this case 10), and the predicted label is the node that has the max value. *Hint: In PyTorch the fully connected layers are called `torch.nn.Linear()`.

**Use these training parameters:** When you train a model, train for 100 epochs with batch size of 10 and use cross entropy loss. In PyTorch's `CrossEntropyLoss` class, the [softmax operation is built in](https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss), therefore you do not need to add a softmax function to the output layer of the network. Use the SGD optimizer with a learning rate of 0.01. 

**Making training sets:** Create training datasets of each of these sizes {500, 1000, 1500, 2000} from MNIST. Note that you should be selecting examples in such a way that you minimize bias, i.e., make sure all ten digits are equally represented in each of your training sets. To do this, you can use `load_mnist_data` function in `load_data.py` where you can adjust the number of examples per digit and the amount of training / testing data. 

*Hint: To read your MNIST dataset for training, you need to use a PyTorch `DataLoader`. To do so, you should use a custom PyTorch `Dataset` class. We included the class definition for you in the HW (`MyDataset` in `data/my_dataset.py`) You can see more details about using custom dataset in this [blog](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) or [github repo](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)). When creating a `DataLoader` set  the `shuffle` property to `True`. 

**Train one model per training set:** Train a new model for each MNIST training set you created and test it on a subset of the MNIST testing set (1000 samples). Use the same architecture for every model. For each model you train, record the loss function value every epoch. Record the time required to train for 100 epochs. From python's built in `time` module, use `time.time()`.

1. (0.25 points) Given the data from your 4 trained models, create a graph that shows the amount of training time along the y-axis and number of training examples along the x-axis.


2. (0.25 points) What happens to your training time as the number of training examples increases? Roughly how many hours would you expect it to take to train on the full MNIST training set using the same architecture on the same hardware you used to create the graph in question 1?


3. (0.25 points) Create a graph that shows classification accuracy on your testing set on the y-axis and number of training 
examples on the x-axis.


4. (0.25 points) What happens to the accuracy as the number of training examples increases?


## Exploring DogSet (.5 points)

DogSet is a subset from a popular machine learning dataset called ImageNet (more info [here](http://www.image-net.org/) and [here](https://en.wikipedia.org/wiki/ImageNet)) which is used for image classification. The DogSet dataset is available [here](https://drive.google.com/open?id=1sKqMO7FwT_DyxQJh1YaMKAI6-lrlMXs0). (Note: you need to be signed into your `@u.northwestern.edu` google account to view this link). As its name implies, the entire dataset is comprised of images of dogs and labels indicating what dog breed is in the image. The metadata, which correlates any particular image with its label and partition, is provided in a file called `dogs.csv`. We have provided a general data loader for you (in `data/dogs.py`), but you may need to adopt it to your needs when using PyTorch. **Note: You need to use the dataset class we provided in MNIST questions to be able to use a PyTorch `DataLoader`**

**Validation sets:** Thus far, you have only used "train" and "test" sets. But it is common to use a third partition called a "validation" set. The validation set is used during training to determine how well a model generalizes to unseen data. The model does *not* train on examples in the validation set, but periodically predicts values in the validation set while training on the training set. Diminishing performance on the validation set is used as an early stopping criterion for the training stage. Only after training has stopped is the testing set used. Here's what this looks like in the context of neural networks: for each epoch a model trains on every example in the training partition, when the epoch is finished the model makes predictions for all of the examples in the validation set and a loss is computed. If the difference between the calculated loss for this iteration and the previous is below some _epsilon_ for _N_ number of epochs in a row, then training stops and we move onto the testing phase.  

5. (0.25 points) In Dogset, how many are in the `train` partition, the `valid` partition and the `test` partition? What is the color palette of the images (greyscale, black & white, RBG)? How many dog breeds are there? 


6. (0.25 points) Select one type of breed. Look through variants of images of this dog breed. Show 3 different images of the same breed that you think are particularly challenging for a classifier to get correct. Explain why you think these three images might be challenging for a classifier. 


## Training a model on DogSet (2 points)

**Build the model architecture:** Create a neural network with two fully connected (aka, dense) hidden layers of size 128, and 64, respectively. Note that you should be flattening your _NxNxC_ image to 1D for your input layer (where _N_ is the height/width, and _C_ is the number of color channels). Your network should have a total of four layers: an input layer that takes in examples, two hidden layers, and an output layer that outputs a predicted class (one node for each dog class in DogSet). Your hidden layers should have a ReLU activation function. Your last (output) layer should be a linear layer with one node per class, and the predicted label is the node that has the max value.

**Use these training parameters:** Use a batch size of 10 and the cross entropy loss. Use the SGD optimizer with a learning rate of 1e-5. 

**When to stop training:** Stop training after 100 epochs or when the validation loss decreases by less than 1e-4, whichever happens first. 

**Training, Testing, Validation sets:** You should use training examples from `train` partition of DogSet. Validation should come from the `valid` partition and testing examples should come from the `test` partition.

7. (0.5 points) How many connections (weights) does this network have?


8. (1.0 point) Train a model on DogSet. After every epoch, record four things: the loss of your model on the training set, the loss of your model on the validation set, and the accuracy of your model on the training and validation sets. 

    * Report the number of epochs your model trained, before terminating.
  
    * Make a graph that has both training and validation loss on the y-axis and epoch on the x-axis.
  
    * Make a graph that has both training and validation accuracy on the y-axis and epoch on the x-axis. 

    * Report the accuracy of your model on the testing set.



9. (0.5 points) Describe the interaction between training loss, validation loss and validation accuracy. When do you think your network stopped learning something meaningful to the problem? Why do you think that? Back up your answer by referring to your graphs.


## Convolutional layers (2 points)

Convolutional layers are layers that sweep over and subsample their input in order to represent complex structures in the input layers. For more information about how they work, [see this blog post](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/). Don't forget to read the PyTorch documentation about Convolutional Layers (linked above).

10. (0.5 points) Convolutional layers produce outputs that are of different size than their input by representing more than one input pixel with each node. If a 2D convolutional layer has `3` channels, batch size `16`, input size `(32, 32)`, padding `(4, 8)`, dilation `(1, 1)`, kernel size `(8, 4)`, and stride `(2, 2)`, what is the output size of the layer?


11. (0.5 point) Combining convolutional layers with fully connected layers can provide a boon in scenarios involving learning from images. Using a similar architecture to the one used in question 8, replace each of your first two hidden layers with a convolutional layer, and add a fully connected layer to output predictions as before. The number of filters (out_channels) should be 16 for the first convolutional layer and 32 for the second convolutional layer. When you call the PyTorch convolutional layer function, leave all of the arguments to their default settings except for kernel size and stride. Determine reasonable values of kernel size and stride for each layer and report what you chose. Tell us how many connections (weights) this network has.


12. (1 point) Train your convolutional model on DogSet. After every epoch, record four things: the loss of your model on the training set, the loss of your model on the validation set, and the accuracy of your model on both training and validation sets. 

    * Report the number of epochs your model trained, before terminating.
  
    * Make a graph that has both training and validation loss on the y-axis and epoch on the x-axis.
  
    * Make a graph that has both training and validation accuracy on the y-axis and epoch on the x-axis. 

    * Report the accuracy of your model on the testing set.


## Digging more deeply into convolutional networks (2 points) ##

The most important property of convolutional networks is their capability in capturing **shift invariant** patterns. You will investigate this property by training a convolutional network to classify simple synthesized images and visualizing the learned kernels. 

**Exploring the synthesized dataset:** the synth_data file included in the /data directory contains 10000 images of simple patterns, divided into 2 classes (5000 images per class). Use the `load_synth_data` function in `data/load_data.py` to load the training features (images) and labels. 

13. (1 point) Go through a few images and plot two examples (1 from each class). What is the common feature among the samples included in each class? What is different from one sample to the next in each class? What information must a classifier rely on to be able to tell these classes apart?


**Build the classifier:** Create a convolutional neural network including three convolutional layers and a linear output layer. The numbers and sizes of filters should be as follows:

* First layer: 2 filters of size (5,5)

* Second layer: 4 filters of size (3,3)

* Third layer: 8 filters of size (3,3)

Use strides of size (1,1) and ReLU activation functions in all convolutional layers. Each convolutional layer should be followed by max-pooling with a kernel size of 2. Use an output linear layer with two nodes, one for each class (note that for binary classification you can use a single node with a sigmoid activation function and binary cross entropy loss, but using softmax and cross entropy keeps the code simpler in this homework).

**Training parameters:** Use a cross entropy loss and the SGD optimizer. Set the batch size to 50 and learning rate to 1e-4. Train the network for 50 epochs.   

14. (1 point) Once the network is trained extract and plot the weights of the two kernels in the first layer. Do these kernels present any particular patterns? If so, what are those patterns and how are they related to the classification task at hand and the classifier performance? Note that since the model is randomly initialized (by default in PyTorch), the shape of kernels might be different across different training sessions. Repeat the experiment a few times and give a brief description of your observations.


## Thinking about deep models (1.5 points)

15. (0.25 points) For any binary function of binary inputs, is it possible to construct some deep network built using only perceptron activation functions that can calculate this function correctly? If so, how would you do it? If not, why not?


17. (0.25 points) Is it possible to learn any arbitrary binary function from data using a network build only using linear activation functions? If so, how would you do it? If not, why not? 

18. (1 point) An adversarial example is an example that is designed to cause your machine learning model to fail. Gradient descent ML methods (like deep networks) update their weights by descending the gradient on the loss function L(X,Y,W) with respect to W. Here, X is a training example, Y is the true label and W are the weights. Explain how you could create an adversarial example by using the gradient with respect to X instead of W.



