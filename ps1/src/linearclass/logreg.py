import numpy as np
from matplotlib import pyplot as plt

import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    fig, ax = plt.subplots()
    ax.scatter(x_eval[:, 1][predictions <= 0.5], x_eval[:, 2][predictions <= 0.5], marker='o')
    ax.scatter(x_eval[:, 1][predictions >= 0.5], x_eval[:, 2][predictions >= 0.5], marker='v')
    ax.plot(np.arange(-10.0, 10.0, 0.1))
    plt.show()
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        for i in range(self.max_iter):
            old_theta = self.theta.copy()
            self.theta -= self.j(x, y) / self.j_prime(x, y)
            if self.verbose:
                print(f'Step {i + 1}: {self.j(x, y)}')
            if sum(abs(self.theta - old_theta)) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(self.theta, x.transpose())
        # *** END CODE HERE ***

    def j(self, x, y):
        return -1 / x.shape[0] * sum(y * np.log(self.h(x)) + (1 - y) * np.log(1 - self.h(x)))

    def j_prime(self, x, y):
        return -1 / x.shape[0] * sum((y * x.transpose() - self.h(x) * x.transpose()).transpose())

    def h(self, x):
        return 1 / (1 + np.exp(np.dot(self.theta, x.transpose())))


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
