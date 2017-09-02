import random
import numpy as np
#import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def find_knn(p, points, k = 5):
    """
    Find the k nearest neighbours of point p and return their indices.
    """
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = np.linalg.norm(p - points[i])
    ind = np.argsort(distances)
    return ind[:k]

def majority(elements):
    """
    Return the most common element in elements
    """
    counts = {}
    for element in elements:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    majority_elements = []
    max_count = max(counts.values())
    for element, count in counts.items():
        if count == max_count:
            majority_elements.append(element)
    return random.choice(majority_elements)
    #mode, count = ss.mstats.mode(elements)
    #return mode

def knn_predict(p, points, outcomes, k = 5):
    """
    Call find_knn() to find the k nearest neighbours of p
    and then call majority() to determine the dominant class
    of the k nearest neighbours before returning the result
    """
    ind = find_knn(p, points, k)
    return majority(outcomes[ind])

def make_prediction_grid(predictors, outcomes, h, k, mode):
    """
    Make a prediction grid using either the homebrew or the scikit-learn knn method
    """
    (x_min, x_max, y_min, y_max) = (predictors[:, 0].min() * 0.9,
                                    predictors[:, 0].max() * 1.1,
                                    predictors[:, 1].min() * 0.9,
                                    predictors[:, 1].max() * 1.1)
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    if mode == "sk":
        prediction_grid = np.zeros(xx.shape, dtype=int)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p = np.array([x, y])
                prediction_grid[j, i] = knn.predict([[x, y]])
        return (xx, yy, prediction_grid)
    elif mode == "homebrew":
        prediction_grid = np.zeros(xx.shape, dtype = int)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                p = np.array([x, y])
                prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
        return (xx, yy, prediction_grid)

# Load the Iris flower data set by Ronald Fisher and
# More on the dataset at https://en.wikipedia.org/wiki/Iris_flower_data_set
iris = datasets.load_iris()


predictors = iris.data[:, 0:2]
outcomes = iris.target

# Determine the accuracy of knn classification on the Iris flower dataset
# with different value of k using both the homebrew and scikit-learn knn methods.
xr = np.array(range(iris.data.shape[0] - 1))
yr_homebrew = np.zeros(xr.shape, dtype = float)
yr_sk  = np.zeros(xr.shape, dtype = float)
for n in xr:
    yr_homebrew[n] = 100 * np.mean(np.array(
        [knn_predict(p, predictors, outcomes, n + 1) for p in predictors]) == outcomes).item()

    knn = KNeighborsClassifier(n_neighbors = n + 1)
    knn.fit(predictors, outcomes)
    yr_sk[n] = 100 * np.mean(knn.predict(predictors) == outcomes).item()

# Use the value of k with the highest accuracy to make prediction grid
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, 0.1,
                                                 yr_homebrew.argmax() + 1, "homebrew")
knn = KNeighborsClassifier(n_neighbors = yr_sk.argmax() + 1)
knn.fit(predictors, outcomes)
(sk_xx, sk_yy, sk_prediction_grid) = make_prediction_grid(predictors, outcomes, 0.1,
                                                          yr_sk.argmax() + 1, "sk")

# Plot the prediction grid along with the Iris flower data points
# using both the homebrew and scikit-learn knn methods for comparison.
# Plot also the accurracy of both the homebrew and scikit-learn knn methods as a function of k.
# Save the plot as knn_iris_plot.pdf in the working direction.
plt.figure(figsize = (12, 12))
grid_cm = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
plt.subplot(221)
plt.title("Homebrew K-Nearest Neighbors with K = " + str(yr_homebrew.argmax() + 1) + "\n" +
          "(Accuracy = " + str(round(100 * np.mean(np.array([knn_predict(p, predictors, outcomes,
                                                                         yr_homebrew.argmax() + 1)
                                                             for p in predictors]) == outcomes).item(), 2)) + ")")
plt.pcolormesh(xx, yy, prediction_grid, cmap = grid_cm, alpha = 0.5)
plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro", label="Iris Setosa")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "bo", label="Iris Virginica")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "go", label="Iris Versicolor")
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.subplot(222)
plt.title("Scikit-learn K-Nearest Neighbors with K = " + str(yr_sk.argmax() + 1) + "\n" + "(Accuracy = " +
          str(round(100 * np.mean(knn.predict(predictors) == outcomes).item(), 2)) + ")")
plt.pcolormesh(sk_xx, sk_yy, sk_prediction_grid, cmap = grid_cm, alpha = 0.5)
plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro", label="Iris Setosa")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "bo", label="Iris Virginica")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "go", label="Iris Versicolor")
plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.subplot(223)
plt.title("Homebrew K-Nearest Neighbors Accuracy (%)")
plt.plot(xr + 1, yr_homebrew, "b-", label="Accuracy (%)")
plt.xlabel('K')
plt.ylabel('Accuracy (%)')
plt.subplot(224)
plt.title("Scikit-learn K-Nearest Neighbors Accuracy")
plt.plot(xr + 1, yr_sk, "b-", label="Accuracy (%)")
plt.xlabel('K')
plt.ylabel('Accuracy (%)')
plt.savefig("knn-sklearn-iris-plot.pdf")
plt.show()
