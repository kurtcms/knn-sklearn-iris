import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

class knn_homebrew_sklearn:
    def __init__(self):
        '''
        Load the Iris flower data set by Ronald Fisher
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset
        '''
        self.iris = datasets.load_iris()
        self.predictors = self.iris.data[:, 0:2]
        self.outcomes = self.iris.target

    def __find_knn(self, p, points, k = 5):
        '''
        Find the k nearest neighbours of point p and return
        their indices
        '''
        distances = np.zeros(points.shape[0])
        for i in range(len(distances)):
            distances[i] = np.linalg.norm(p - points[i])
        ind = np.argsort(distances)
        return ind[:k]

    def __majority(self, elements):
        '''
        Return the most common element in elements
        '''
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

    def _knn_predict(self, p, points, outcomes, k = 5):
        '''
        Call __find_knn() to find the k nearest neighbours of p
        and then call __majority() to determine the dominant class
        of the k nearest neighbours before returning the result
        '''
        ind = self.__find_knn(p, points, k)
        return self.__majority(outcomes[ind])

    def _make_prediction_grid(self, predictors, outcomes, h, k, mode):
        '''
        Make a prediction grid using either the homebrew or
        the scikit-learn knn method
        '''
        (x_min, x_max, y_min, y_max) = (predictors[:, 0].min() * 0.9,
                                        predictors[:, 0].max() * 1.1,
                                        predictors[:, 1].min() * 0.9,
                                        predictors[:, 1].max() * 1.1)
        xs = np.arange(x_min, x_max, h)
        ys = np.arange(y_min, y_max, h)
        xx, yy = np.meshgrid(xs, ys)

        if mode == 'sk':
            prediction_grid = np.zeros(xx.shape, dtype=int)
            knn = KNeighborsClassifier(n_neighbors = k)
            knn.fit(predictors, outcomes)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p = np.array([x, y])
                    prediction_grid[j, i] = knn.predict([[x, y]])
            return (xx, yy, prediction_grid)
        elif mode == 'homebrew':
            prediction_grid = np.zeros(xx.shape, dtype = int)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p = np.array([x, y])
                    prediction_grid[j, i] = self._knn_predict(p,
                    predictors, outcomes, k)
            return (xx, yy, prediction_grid)

    def predict_by_k(self):
        '''
        Determine the accuracy of knn classification on the
        Iris flower dataset with different value of k using
        both the homebrew and scikit-learn knn methods
        '''
        self.xr = np.array(range(self.iris.data.shape[0] - 1))
        self.yr_homebrew = np.zeros(self.xr.shape, dtype = float)
        self.yr_sk  = np.zeros(self.xr.shape, dtype = float)
        for n in self.xr:
            self.yr_homebrew[n] = 100 * np.mean(np.array(
                [self._knn_predict(p, self.predictors, self.outcomes,
                n + 1) for p in self.predictors]) == self.outcomes).item()

            knn = KNeighborsClassifier(n_neighbors = n + 1)
            knn.fit(self.predictors, self.outcomes)
            self.yr_sk[n] = 100 * np.mean(knn.predict(self.predictors)
             == self.outcomes).item()

    def make_prediction_grid_with_highest_k(self):
        '''
        Use the value of k with the highest accuracy to make prediction grid
        '''
        (self.xx, self.yy, self.prediction_grid) = self._make_prediction_grid(
                                                    self.predictors, self.outcomes,
                                                    0.1, self.yr_homebrew.argmax()
                                                    + 1, 'homebrew')
        self.knn = KNeighborsClassifier(n_neighbors = self.yr_sk.argmax() + 1)
        self.knn.fit(self.predictors, self.outcomes)
        (self.sk_xx, self.sk_yy, self.sk_prediction_grid) = self._make_prediction_grid(
                                                            self.predictors, self.outcomes,
                                                            0.1, self.yr_sk.argmax() + 1, 'sk')

    def plot_and_save_pdf(self):
        '''
        Plot the prediction grid with the Iris flower data set
        using both the homebrew and scikit-learn knn methods for comparison.
        Plot also the accurracy of both the homebrew and scikit-learn knn
        methods as a function of k.
        Save the plot as knn_iris_plot.pdf in the working directory.
        '''
        plt.figure(figsize = (12, 12))
        grid_cm = ListedColormap (['hotpink','lightskyblue', 'yellowgreen'])
        plt.subplot(221)
        plt.title('Homebrew K-Nearest Neighbors with K = ' +
                    str(self.yr_homebrew.argmax() + 1) + '\n' +
                    '(Accuracy = ' + str(round(100 * np.mean(np.array(
                    [self._knn_predict(p, self.predictors, self.outcomes,
                    self.yr_homebrew.argmax() + 1) for p in self.predictors])
                    == self.outcomes).item(), 2)) + ')')
        plt.pcolormesh(self.xx, self.yy, self.prediction_grid, cmap = grid_cm, alpha = 0.5)
        plt.plot(self.predictors[self.outcomes == 0][:, 0],
            self.predictors[self.outcomes == 0][:, 1], 'ro', label='Iris Setosa')
        plt.plot(self.predictors[self.outcomes == 1][:, 0],
            self.predictors[self.outcomes == 1][:, 1], 'bo', label='Iris Virginica')
        plt.plot(self.predictors[self.outcomes == 2][:, 0],
            self.predictors[self.outcomes == 2][:, 1], 'go', label='Iris Versicolor')
        plt.legend()
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.subplot(222)
        plt.title('Scikit-learn K-Nearest Neighbors with K = ' +
                    str(self.yr_sk.argmax() + 1) + '\n' + '(Accuracy = ' +
                    str(round(100 * np.mean(self.knn.predict(self.predictors)
                    == self.outcomes).item(), 2)) + ')')
        plt.pcolormesh(self.sk_xx, self.sk_yy, self.sk_prediction_grid, cmap = grid_cm, alpha = 0.5)
        plt.plot(self.predictors[self.outcomes == 0][:, 0],
            self.predictors[self.outcomes == 0][:, 1], 'ro', label='Iris Setosa')
        plt.plot(self.predictors[self.outcomes == 1][:, 0],
            self.predictors[self.outcomes == 1][:, 1], 'bo', label='Iris Virginica')
        plt.plot(self.predictors[self.outcomes == 2][:, 0],
            self.predictors[self.outcomes == 2][:, 1], 'go', label='Iris Versicolor')
        plt.legend()
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.subplot(223)
        plt.title('Homebrew K-Nearest Neighbors Accuracy (%)')
        plt.plot(self.xr + 1, self.yr_homebrew, 'b-', label='Accuracy (%)')
        plt.xlabel('K')
        plt.ylabel('Accuracy (%)')
        plt.subplot(224)
        plt.title('Scikit-learn K-Nearest Neighbors Accuracy')
        plt.plot(self.xr + 1, self.yr_sk, 'b-', label='Accuracy (%)')
        plt.xlabel('K')
        plt.ylabel('Accuracy (%)')
        plt.savefig('knn-sklearn-iris-plot.pdf')
        plt.show()

if __name__ == '__main__':
    '''
    Create the knn object and plot the prediction grid to compare
    the homebrew and scikit-learn knn methods on the Iris flower data set.
    Save the plot in the working directory.
    '''
    iris_knn = knn_homebrew_sklearn()
    iris_knn.predict_by_k()
    iris_knn.make_prediction_grid_with_highest_k()
    iris_knn.plot_and_save_pdf()
