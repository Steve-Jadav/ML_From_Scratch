import numpy as np
import random
from collections import Counter

class LogisticRegression():

    def __init__(self, learning_rate: float, epochs: int, fit_intercept: bool = False,
                verbose: bool = False):

        """
        :param learning_rate: float
        :param epochs: int :- Number of iterations for the optimization algorithm.
        :param fit_intercept: bool :- Fits an intercept if set to True. """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.model_parameters = None


    def __sigmoid(self, z: np.ndarray):
        return (1 / (1 + np.exp(-z)))


    def __loss(self, hx: np.ndarray, y: np.ndarray):
        return (-y * np.log(hx) - (1 - y) * np.log(1 - hx)).mean()


    def __gradient_descent(self, X_train: np.ndarray, y_train: np.ndarray, parameters: np.ndarray):

        """
        Optimization algorithm which tries to minimize the convex cost function.
        :param X_train: np.ndarray Training features
        :param y_train: np.ndarray Training class labels
        :param parameters: np.ndarray Theta vector """

        m, n_features = X_train.shape[0], X_train.shape[1]

        for i in range(0, self.epochs):
            z = np.dot(X_train, parameters)
            hx = self.__sigmoid(z)
            gradient = np.dot(X_train.T, (hx - y_train[0])) / m
            parameters = parameters - (self.learning_rate * gradient)

            if(i % 2000 == 0 and self.verbose):
                z = np.dot(X_train, parameters)
                hx = self.__sigmoid(z)
                print(f'loss: {self.__loss(hx, y_train[0])} \t')

        return parameters


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        """
        :param X_train: np.ndarray
        :param y_train: np.ndarray
        :return: None """

        m = X_train.shape[0]
        if self.fit_intercept:
            x0 = np.ones((m, 1))
            X_train = np.hstack((x0, X_train)) # Features
        n_features = X_train.shape[1]
        parameters = np.zeros(n_features)  # Theta vector
        self.model_parameters = self.__gradient_descent(X_train, y_train, parameters)

        print ('Model Trained with parameters: ', self.model_parameters)


    def predict(self, X_test: np.ndarray):

        """ Given a test data set, returns the class predictions. """

        m = X_test.shape[0]
        if self.fit_intercept:
            x0 = np.ones((m, 1))
            X_test = np.hstack((x0, X_test))
        n_features = X_test.shape[1]
        predictions = list()

        probabilities = self.__sigmoid(np.dot(X_test, self.model_parameters))

        for value in probabilities:
            if value <= 0.5: predictions.append(0)
            else: predictions.append(1)

        return predictions


    def score(self, y_predictions: np.ndarray, y_true: np.ndarray):

        """
        :param y_predictions: np.ndarray
        :param y_true: np.ndarray
        :return: accuracy of the Logistic Regression model """

        return float(sum(y_predictions == y_true[0])) / float(len(y_true[0])) * 100


class NaiveBayes():

    def __init__(self):
        self.mean = None
        self.variance = None
        self.priors = None

    def __calculate_prior(self, y_train: np.ndarray):

        """ Calculates the prior probabilites i.e. P(class)
        :param: y_train
        :return prior_probabilites for each class label: np.ndarray """

        labels = Counter(y_train[0])
        prior_probabilites = np.zeros(2)
        for i in range(0, 2):
            prior_probabilites[i] = labels.get(i)/y_train.shape[1]

        return prior_probabilites


    def __calculate_likelihood(self, X_test: np.ndarray):

        """ Calculates the likelihood of X given c, i.e. P(X|c)
        :param: X_test: test dataset
        :return: posterior probabilites of each feature given class: np.ndarray """

        n_features = X_test.shape[1]
        m = X_test.shape[0]
        posteriors = np.zeros((m, 2))

        for z in range(0, m):
            for i in range(0, 2):
                L = 1
                for j in range(0, n_features):
                    L = L * (1/np.sqrt(2*3.14*self.variance[i][j])) * \
                    np.exp(-0.5 * pow((X_test[z][j] - self.mean[i][j]), 2)/self.variance[i][j])
                posteriors[z][i] = L

        return posteriors


    def __mean_variance(self, X_train: np.ndarray, y_train: np.ndarray):

        """
        :param X_train: np.ndarray
        :param y_train: np.ndarray
        :return: mean, variance (np.ndarray) of each feature for each class """

        m = X_train.shape[0]
        n_features = X_train.shape[1]
        mean = np.zeros((2, n_features))
        variance = np.zeros((2, n_features))
        temp0 = np.zeros((0, n_features))
        temp1 = np.zeros((0, n_features))

        for i in range(0, m):
            if y_train[0][i] == 0:
                temp0 = np.append(temp0, X_train[i].reshape(1,2), axis=0)
            else:
                temp1 = np.append(temp1, X_train[i].reshape(1,2), axis=0)

        for i in range(0, n_features):
            mean[0][i] = np.mean(temp0[:][:,i])
            variance[0][i] = np.var(temp0[:][:,i])
            mean[1][i] = np.mean(temp1[:][:,i])
            variance[1][i] = np.var(temp1[:][:,i])

        return mean, variance


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        """
        :param X_train: np.ndarray
        :param y_train: np.ndarray
        :return: None """

        self.priors = self.__calculate_prior(y_train)
        self.mean, self.variance = self.__mean_variance(X_train, y_train)


    def predict(self, X_test: np.ndarray):

        """ Given a test data set, returns the class predictions. """

        posteriors = self.__calculate_likelihood(X_test)
        m = X_test.shape[0]
        conditional_probabilities = np.zeros(2)
        predictions = np.zeros(m)

        for i in range(0, m):
            total_probability = 0

            for j in range(0, 2):
                total_probability += (posteriors[i][j] * self.priors[j])

            for j in range(0, 2):
                conditional_probabilities[j] = (posteriors[i][j] * self.priors[j])/total_probability

            predictions[i] = float(conditional_probabilities.argmax())

        return predictions


    def score(self, y_predictions: np.ndarray, y_true: np.ndarray):

        """
        :param y_predictions: np.ndarray
        :param y_true: np.ndarray
        :return: accuracy of the NB model """

        return float(sum(y_predictions == y_true[0])) / float(len(y_true[0])) * 100

class KMeans():

    def __init__(self, n_clusters: int, max_k_means: bool):

        """
        :param n_clusters: int :- Number of clusters to fit on the dataset
        :param max_k_means: bool :- Boolean variable indicating a different approach to select
                                    initial cluster centroid  """

        self.n_clusters = n_clusters
        self.max_k_means = max_k_means
        self.centroids = None
        self.cluster_assignments = dict()


    def __objective_function(self, centroids: np.ndarray, cluster_assignments: dict):

        """
        :param centroids: np.ndarray :- Cluster centroid vector
        :param cluster_assignments: dict :- Hashmap mapping cluster to its points
        :output: Returns the sum of squared errors (SSE) """

        cost: float = 0.0

        for cluster in range(0, self.n_clusters):
            squared_error = (cluster_assignments[cluster] - centroids[cluster]) ** 2
            cost += np.sum(squared_error)

        return cost


    def __plot_clusters(self, centroids: np.ndarray, cluster_assignments: dict, K: int):

        """
        :param centroids: np.ndarray :- Cluster centroid vector
        :param cluster_assignments: dict :- Hashmap mapping cluster to its points
        :param K: int :- Number of clusters """

        centX = centroids[:,0]
        centY = centroids[:,1]
        colors = ["Red", "Blue", "Green", "Purple", "Cyan", "Orange", "Grey"]
        plot_name = "Figure_" + str(K)
        plt.clf()

        for i in range(0, K):
            X = cluster_assignments[i][:,0]
            Y = cluster_assignments[i][:,1]
            sns.scatterplot(X, Y, marker = "p", alpha = 0.6, color = colors[i], label = 'Cluster ' + str(i))

        sns.scatterplot(centX, centY, marker = "*", alpha = 1, s = 170, color = "Black", label = "Centroids")
        plt.title(str(K) + ' Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig(plot_name)


    def fit(self, X: np.ndarray):

        """
        :param X: np.ndarray: n-dimensional input dataset """

        if self.n_clusters == 0:
            print ('Number of clusters need to be atleast 1.')
            return

        n = X.shape[0]      # Number of instances
        m = X.shape[1]      # Number of features

        self.centroids = np.array([]).reshape(0, m)

        if self.max_k_means == False:

            # Compute random initial cluster centroids
            for i in range(0, self.n_clusters):
                rand = random.randint(0, n-1)
                self.centroids = np.append(self.centroids, X[rand].reshape(1, m), axis = 0)

        else:

            # The first center is random
            rand = random.randint(0, n-1)
            self.centroids = np.append(self.centroids, X[rand].reshape(1, m), axis = 0)

            # The remaining centers should not be random.
            # Approach: Select the samples which are farthest from the previous i-1 centers.
            for i in range(1, self.n_clusters):
                distances = np.zeros(n)
                for j in range(0, i):
                    distances += np.sqrt(np.sum((X - self.centroids[j]) ** 2, axis=1))

                distances = np.divide(distances, i)
                max_instance = np.argmax(distances)
                self.centroids = np.append(self.centroids, X[max_instance].reshape(1, m), axis = 0)


        while True:

            # Initial empty cluster assignments
            for cluster in range(0, self.n_clusters):
                self.cluster_assignments[cluster] = np.array([]).reshape(0, m)

            for i in range(0, n):
                # Calculate the distance between current point and all the centroids
                distances = np.sqrt(np.sum((X[i] - self.centroids)**2, axis = 1))

                # Select the centroid with the minimum distance
                cluster = np.argmin(distances)
                self.cluster_assignments[cluster] = np.append(self.cluster_assignments[cluster], X[i].reshape(1, m), axis=0)

            # Handling empty cluster assignments
            for i in self.cluster_assignments.keys():
                if len(self.cluster_assignments[i]) == 0:
                    self.cluster_assignments[i] = np.mean(X, axis=0)


            new_centroids = np.zeros((self.centroids.shape[0], self.centroids.shape[1]))

            # Compute the new centroids
            for i in range(0, self.n_clusters):
                new_centroids[i] = np.mean(self.cluster_assignments[i], axis=0)

            if np.array_equal(self.centroids, new_centroids):
                break

            self.centroids = new_centroids


        #print ('Final converged centroids: \n', self.centroids)
        #self.__plot_clusters(self.centroids, self.cluster_assignments, self.n_clusters)
        cost = self.__objective_function(self.centroids, self.cluster_assignments)
        return cost
    

class Preprocessing():

    def __init__(self):
        pass

    def feature_extraction(self, data):
        rows = data.shape[0]
        new_data = np.zeros((rows, 2))

        for i in range(0, rows):
            new_data[i][0] = np.mean(data[i])
            new_data[i][1] = np.std(data[i])

        return new_data
