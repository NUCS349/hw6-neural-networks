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

For this assignment, it may be helpful to know how to save and load trained models to and from disc. See [this tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) about how to do that. We recommend you read all of the questions prior to training your models in case you decide that you want to save your trained models to use for later questions.


## Understanding PyTorch (1 point)

Read the first four tutorials on [this page](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) ("*What is PyTorch?*", "*Autograd: Automatic Differentiation*", "*Neural Networks*", and "*Training a Classifier*") and then answer the following questions.

1. (0.25 points) What is a tensor? How is it different than a matrix?

2. (0.5 points) What is automatic differentiation? How does it relate to gradient descent and back-propagation? Why is it important for PyTorch?

3. (0.25 points) Why does PyTorch have its own tensor class (`torch.Tensor`) when it is extremely similar to numpy's `np.ndarray`?

## Training on MNIST (2 points)

Let us train neural networks to classify handwritten digits from the MNIST dataset and analyze the accuracy and training time of these neural networks. 

**Build the model architecture:** Create a neural network with two fully connected (aka, dense) hidden layers of size 128, and 64, respectively. Your network should have a total of four layers: an input layer that takes in examples, two hidden layers, and an output layer that outputs a predicted class (10 possible classes, one for each digit class in MNIST). Your hidden layers should have a ReLU activation function and your last (output) layer should have a softmax activation function and be of size 10. *Hint: In PyTorch the fully connected layers are called `torch.nn.Linear()`.

*QUESTION: WHAT SHOULD THE OUTPUT LAYER LOOK LIKE? 10 NODES WITH SIGMOID ACTIVATION? WHAT ARE YOU THINKING HERE? (prem: tried to answer)

**Use these training parameters:** When you train a model, train for 100 epochs with batch size of 10 and using cross entropy loss. Use the SGD optimizer with a learning rate of 0.01. 

**Making training sets:** Create one training set of each of these sizes {500, 1000, 1500, 2000} from MNIST. Note that you should be selecting examples in such a way that you minimize bias, i.e., make sure all ten digits are equally represented in each of your training sets. To do this, you can use `load_mnist_data` function in `load_data.py` where you can adjust the number of examples per digit and the amount of training / testing data. 

*Hint: To read your MNIST dataset for training, you may not need to use a PyTorch `DataLoader`. If, however, you want to use it with your numpy NMIST dataset, you should use your custom dataset class. We included the class definition for you in the HW (`MyDataset` in `my_dataset.py`) You can see more details about using custom dataset in this [blog](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) or [github repo](https://github.com/utkuozbulak/pytorch-custom-dataset-examples))

**Train one model per training set:** Train a new model for each MNIST training set you created and test it on the MNIST testing subset. Use the same architecture for every model. For each model you train, record the loss function value every epoch. Record the time required to train for 100 epochs. Once the model is trained, record testing accuracy on the training set using the model.

4. (0.5 points) Given the data from your 4 trained models, create a graph that shows the amount of training time along the y-axis and number of training examples along the x-axis. 

5. (0.5 points) What happens to your training time as the number of training examples increases? Roughly how many hours would you expect it to take to train on the full MNIST training set using the same architecture on the same hardware you used to create the graph in question 4?

6. (0.5 points) Create a graph that shows classification accuracy on your testing set on the y-axis and number of training 
examples on along the x-axis. 

7. (0.5 points) What happens to the accuracy as the number of training examples increases?

## DogSet (2.5 points)

DogSet is a subset from a popular machine learning dataset called ImageNet (more info [here](http://www.image-net.org/) and [here](https://en.wikipedia.org/wiki/ImageNet)) which is used for image classification. The DogSet dataset is available [here](https://drive.google.com/open?id=1wlZZ8MBbcugcmiPqB4QJ8Jh9BdVFJSoC). (Note: you need to be signed into your `@u.northwestern.edu` google account to view this link). As it name implies, the entire dataset is comprised of images of dogs and labels indicating what dog breed is in the image. The metadata, which correlates any particular image with its label and partition, is provided in a file called `dogs.csv`. We have provided a general data loader for you (in `data/dogs.py`), but you may need to adopt it to your needs when using PyTorch. **Note: You may need to use the dataset class we provided in MNIST questions if you want to use a PyTorch `DataLoader`**

**Validation sets:** Thus far, you have only used "train" and "test" sets. But it is common to use a third partition called a "validation" set. The validation set is used during training to determine how well a model generalizes to unseen data. The model does *not* train on examples in the validation set, but periodically predicts values in the validation set while training on the training set. Diminishing performance on the validation set is used as an early stopping criterion for the training stage. Only after training has stopped is the testing set used. Here's what this looks like in the context of neural networks: for each epoch a model trains on every example in the training partition, when the epoch is finished the model makes predictions for all of the examples in the validation set and a loss is computed. If the difference between the calculated loss for this iteration and the previous is below some _epsilon_ for _N_ number of epochs in a row, then training stops and we move onto the testing phase.  

The goal of the next three questions is to analyze DogSet and train neural networks to classify dog breeds.

8. (0.5 points) How many images are in DogSet total? How many have labels? How many are in the `train` partition? How many in the `valid` partition? How many are in the `test` partition? What are the dimensions of each image? What is the color palette of the images (greyscale, black & white, RBG)? How many dog breeds are there? 

*QUESTION: There are 8 parts to a 1/2 point question. How do you plan on grading this?

9. (0.5 points) Select one type of breed. Look through variants of images of this dog breed. Show 3 different images of the same breed that you think are particularly challenging for a classifier to get correct. Explain why you think these three images might be challenging for a classifier. 

==============QUESTION 10 IS BIG==========
10. (1.0 point) You will train and test models on subsets of DogSet. If you look at `dogs.csv`, you will notice that there are many unlabled examples in the testing partition. Therefore, you will use only `train` and `valid` partitions of DogSet. 

Based on your experience with neural networks from the MNIST dataset, roughly determine a subset of the DogSet for training, validation, and testing

*QUESTION: ROUGHLY DETERMINE?

to train a neural network with the same architecture (meaning, a network with two fully connected hidden layers with nodes of size 128 and 64, and ReLU activation functions) and loss that you used for your MNIST networks in questions 5-7. 

You should sample traning examples from `train` partition of DogSet. Validation and testing set should be sampled from `valid` partition of DogSet. 

This network will have a different input size and output number of classes from the network for MNIST dataset. Train the network on your dataset.

Stop training after 100 epochs or your validation loss changes by less than 1e-4 for three epochs in a row, whichever happens first. 

Report the size of your training, validation and testing sets, the number of epochs it trained for and batch size that you used. 

After every epoch, record three things: the loss of your model on the training set, the loss of your model on the validation set, and the accuracy of your model on the validation set. 

*QUESTION: THE TWO QUESTIONS BELOW RE THE ONLY GRADABLE ITEMS. WHAT ARE THE POINT VALUES FOR THESE 2 GRAPHS AND FOR REPORTING ACCURACY?

Make one graph that has both training and validation loss on the y-axis and epoch on the x-axis, and another graph that has the validation accuracy on the y-axis and epoch on the x-axis. 

Report the accuracy of your model on the testing set. **(Note: If you want, you can use all the labeled examples in DogSet dataset, not sampling its subsets. But you still need to split the `valid` partition of DogSet into validation and testing set for your experiment.)**

=========QUESTION 10 ENDS HERE==========

11. (0.5 points) Now, using the same `train`, `valid`, and `test` DogSet subsets as in question 10, train a new network with _three_ fully connected hidden layers of size 256, 128, and 64 respectively (five layers total). As before, the hidden layers should all use a ReLU activation function, the last layer should use softmax, and the loss function is cross entropy. Use the same stopping criterion as question 10. Make the same graphs as in question 10. Report the accuracy of your model on the testing set and how long it took to train. How did the accuracy change? How did the loss change?


## Convolutional Layers, Pooling (2.5 points)

Convolutional layers are layers that sweep over and subsample their input in order to represent complex structures in the input layers. For more information about how they work, [see this blog post](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/). Don't forget to read the PyTorch documentation about Convolutional Layers (linked above).

12. (0.5 points) Convolutional layers produce outputs that are of different size than their input by representing more than one input pixel with each node. If a 2D convolutional layer has `3` channels, batch size `16`, input size `(32, 32)`, padding `(4, 8)`, dilation `(1, 1)`, kernel size `(8, 4)`, and stride `(2, 2)`, what is the output size of the layer?


13. (1 point) Combining convolutional layers with fully connected layers can provide a boon in scenarios involving learning from images. Using your same `train`, `valid`, and `test` DogSet subsets from questions 10 and 11, replace each of your first two hidden layers with a convolutional layer. When you call the PyTorch convolutional layer function, leave all of the arguments to their default settings except for kernel size and stride. 

*QUESTION: IS THE NUMBER OF CHANNELS DEFAULT A REASONABLE ONE? (SEE PREVIOUS PARAGRAPH) ALSO, WHAT ARE THE POINT VALUES FOR REPORTING YOUR KERNEL SIZE VS MAKING THE GRAPH?

Determine reasonable values of kernel size and stride for each layer and report what you chose. Train your neural network with the same training parameters as before (number of epochs, stopping condition, batch size, loss function, learning rate, etc). Make the same graphs as question 10 (train and validation loss, and validation accuracy) and report the results of your classifier on the testing set.

14. (1 point) Pooling can reduce the feature space of convolutional layers while keeping the most important information. A common type of pooling is _max pooling_, where the maximum value of each convolutional filter is kept. Keeping everything the same as question 13, add a max pooling layer after each convolutional layer that you added in question 13. Train and test on the same DogSet subset as before. How did adding max pooling affect your accuracy? Report your accuracy and training time.

*QUESTION: WHAT ARE YOU LOOKING FOR IN THE ANSWERS TO 14? WHAT IS THE WEIGHTING OF THE POINTS?


## Using Pre-Trained Models (1.0 points)

The models you have trained have been comparatively tiny compared to the state-of-the-art neural networks on tasks like ImageNet. PyTorch allows you to download a number of very powerful models pre-trained on many common machine learning tasks. You can see a list of all available models trained on ImageNet [here](https://pytorch.org/docs/stable/torchvision/models.html#classification).

15. (0.5 points) Select 3 models from the list of pretrained models (**WARNING:** some models are as big as ~500MB!) and answer the following questions: How many layers does this model have? How many are convolutional layers? How many are fully connected layers? What do "top-1 error" and "top-5 error" mean? What is the top-1 error for the model you choose? What is the top-5 error?

16. (0.5 points) Now use your selected pretrained models and use them to predict dog breeds from your DogSet `test` partition. Compare the results from each of the pretrained models to the best dog breed classifier model that you trained. **HINT: you may have to upsample the DogSet images to use with the pretrained models. See the docs for the `DogsDataset` class.**

*QUESTION: WHAT'S THE PLAN TO DEAL WITH BONGJUN'S QUESITONS ABOUT THE NON-COMPARABLE NATURE OF THE DOWNLOADED IMAGENET-TRAINED MODELS VS THE DOGNET MODELS?


