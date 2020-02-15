import numpy as np
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
