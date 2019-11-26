import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
	batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
	"""
	This function either trains or evaluates a model. 

	training mode: the model is trained and evaluated on a validation set, if provided. 
				   If no validation set is provided, the training is performed for a fixed 
				   number of epochs. 
				   Otherwise, the model should be evaluted on the validation set 
				   at the end of each epoch and the training should be stopped based on one
				   of these two conditions (whichever happens first): 
				   1. The validation loss stops improving. 
				   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
	learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
    	  The value of each key is a list of loss values. Each loss value is the average
    	  of training/validation loss over one epoch.
    	  If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
    	 The value of each key is a list of accuracies (percentage of correctly classified
    	 samples in the dataset). Each accuracy value is the average of training/validation 
    	 accuracies over one epoch. 
    	 If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
	
	Summary of the operations this function should perform:
	1. Use the DataLoader class to generate trainin, validation, or test data loaders
	2. In the training mode:
	   - define an optimizer (we use SGD in this homework)
	   - call the train function (see below) for a number of epochs untill a stopping
	     criterion is met
	   - call the test function (see below) with the validation data loader at each epoch 
	     if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

	"""

	raise NotImplementedError()


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

	"""
	This function implements ONE EPOCH of training a neural network on a given dataset.
	Example: training the Digit_Classifier on the MNIST dataset


	Inputs:
	model: the neural network to be trained
	data_loader: for loading the netowrk input and targets from the training dataset
	optimizer: the optimiztion method, e.g., SGD 
	device: we run everything on CPU in this homework

	Outputs:
	model: the trained model
	train_loss: average loss value on the entire training dataset
	train_accuracy: average accuracy on the entire training dataset
	"""

	raise NotImplementedError()


def _test(model, data_loader, device=torch.device('cpu')):
	"""
	This function evaluates a trained neural network on a validation set
	or a testing set. 

	Inputs:
	model: trained neural network
	data_loader: for loading the netowrk input and targets from the validation or testing dataset
	device: we run everything on CPU in this homework

	Output:
	test_loss: average loss value on the entire validation or testing dataset 
	test_accuracy: percentage of correctly classified samples in the validation or testing dataset
	"""

	raise NotImplementedError()




