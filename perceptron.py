import numpy as np


class Perceptron:
    """
    Parameters
    ----------
    eta: float
        learning ratio(0.0; 1.0)
    iteration_number: int
        number of circuits on learning packages

    Attributes
    ----------
    weight_vector: vector
        weights
    wrong_class: list
        number of wrong classifications in each circuit
    """
    def __init__(self, eta=0.01, iteration_number=50, random_speed=1):
        self.wrong_class = []
        self.eta = eta
        self.n_iter = iteration_number
        self.random_speed = random_speed

    def fit(self, learning_vectors, result):
        """
        Fitting learning data
        Parameters
        ----------
        learning_vectors: matrix [n_sample, n_feature]
        result: matrix [n_sample]
            target value

        Returns
        -------
        self: object
        """
        rgen = np.random.RandomState(self.random_speed)
        self.weight_vector = rgen.normal(loc=0.0, scale=0.01, size=1 + learning_vectors.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(learning_vectors, result):
                update = self.eta * (target - self.predict(xi))
                self.weight_vector[1:] += update * xi
                self.weight_vector[0] += update
                errors += int(update != 0.0)
            self.wrong_class.append(errors)
        return self

    def net_input(self, learning_vectors):
        return np.dot(learning_vectors, self.weight_vector[1:] + self.weight_vector[0])

    def predict(self, learning_vectors):
        return np.where(self.net_input(learning_vectors) >= 0.0, 1, -1)

