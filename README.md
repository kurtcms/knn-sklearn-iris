# Homebrew k-Nearest Neighbors Algorithm (k-NN) vs scikit-learn's built-in KNeighborsClassifier on the Iris Flower Dataset
This Python script does the following:

1. Builds a homebrew k-nearest neighbors algorithm (k-NN).
2. Uses the homebrew k-NN as well as scikit-learn's built-in KNeighborsClassifier function, to classify the 3 different types of Iris flowers (Setosa, Versicolour, and Virginica), by their sepal length and width, from Ronald Fisher's Iris flower data set (https://en.wikipedia.org/wiki/Iris_flower_data_set), and measures their accuracies, for k equals to 1 to the number of entries in the Iris flower data set minus 1.
3. Plots the Iris flower data set on prediction grids generated by the homebrew k-NN and scikit-learn's KNeighborsClassifier, each using a value of k that maximises their respective prediction accuracies.
4. Plots also the prediction accuracies against k of both the homebrew k-NN and scikit-learn's KNeighborsClassifier for comparison.

![alt text](https://kurtcms.org/git/knn-sklearn-iris/knn-sklearn-iris-plot.png)

## Table of Content

- [Getting Started](#getting-started)
  - [Git Clone](#git-clone)
  - [Dependencies](#dependencies)
  - [Run](#run)

## Getting Started

Get started in three simple steps:

1. [Download](#git-clone) a copy of the script;
2. Install the [dependencies](#dependencies); and
3. [Run](#run) the script manually.

### Git Clone

Download a copy of the script with `git clone`
```shell
$ git clone https://github.com/kurtcms/knn-sklearn-iris /opt/
```

### Dependencies

This script requires the following libraries:

1. [NumPy](https://github.com/numpy/numpy)
2. [Matplotlib](https://github.com/matplotlib/matplotlib)
3. [scikit-learn](https://github.com/scikit-learn/scikit-learn)

Install them with [`pip3`](https://github.com/pypa/pip):

```shell
$ pip3 install numpy matplotlib scikit-learn
```

### Run

Run the script with [`Python 3`](https://github.com/python/cpython)

```shell
$ python3 /opt/knn-sklearn-iris/knn-sklearn-iris.py
```
