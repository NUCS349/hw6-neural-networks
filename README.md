# HW6: Neural Networks for CS349 @ NU
**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's netid 
here, we will check.


In this assignment, you will:
- Learn how to use PyTorch
- Implement two types of neural networks
- Explore the impact of different hyperparameters on your networks' performance

## Clone this repository

To clone this repository install GIT on your computer and copy the link of the repository (find above at "Clone or Download") and enter in the command line:

``git clone YOUR-LINK``

Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Jupyter Notebook

We have created a jupyter notebook which you can use that should guide you through the complete homework. We heavily recommend using it, since most Deep Learning research is now done within Jupyter Notebooks. You find the notebook here:

experiments/deep-learning-experiments.ipynb

## Environment setup

Make a conda environment for this assignment, and then run:

``pip install -r requirements.txt``

**IMPORTANT: PyTorch is not included in `requirements.txt`!** To install PyTorch, find the correct install command for your operating system and version of python [here](https://pytorch.org/get-started/locally/). For "PyTorch Build" select the `Stable (1.1)` build, select your operating system, for "Package" select `pip` , for "Language" select your python version (either python 3.5, 3.6, or 3.7), and finally for "CUDA" select `None`. **Make sure to run the command with your conda environment activated.**


## Running the test cases

The test cases can be run with:

``python -m pytest -s``

at the root directory of the assignment repository.

## Questions? Problems? Issues?

Simply open an issue on the starter code repository for this assignment [here](https://github.com/NUCS349/hw-FILL-IN/issues). Someone from the teaching staff will get back to you through there!
