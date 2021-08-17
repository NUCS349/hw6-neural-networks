# HW8: Neural Networks for EECS 349 @ NU
**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's netid 
here, we will check.


In this assignment, you will:
- Learn about PyTorch
- Implement two neural networks

## Clone this repository

To clone this repository run the following command:

``git clone https://github.com/nucs349/hw1-FILL-IN-[your_username]``

`[your_username]` is replaced in the above link by your Github username. Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Environment setup

Make a conda environment for this assignment, and then run:

``pip install -r requirements.txt``

**Important: PyTorch is not included in `requirements.txt`!** To install PyTorch, find the correct install command for your operating system and version of python [here](https://pytorch.org/get-started/locally/). For "PyTorch Build" select the `Stable (1.1)` build, select your operating system, for "Package" select `pip` , for "Language" select your python version (either python 3.5, 3.6, or 3.7), and finally for "CUDA" select `None`. **Make sure to run the command with your conda environment activated**

_Note: To determine which python you are using, type `python` into your command line to get an interactive shell, you should see your python version in the first line._

## Running the test cases

The test cases can be run with:

``python -m pytest -s``

at the root directory of the assignment repository.

## Questions? Problems? Issues?

Simply open an issue on the starter code repository for this assignment [here](https://github.com/NUCS349/hw-FILL-IN/issues). Someone from the teaching staff will get back to you through there!