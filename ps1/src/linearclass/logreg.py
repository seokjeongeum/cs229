import numpy as np
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
    np.savetxt(save_path, clf.predict(x_eval))
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
            old_theta = self.theta
            self.theta -= self.j(x, y) / self.j_prime(x, y)
            if self.verbose:
                print(f'Step {i}: {self.j(x, y)}')
            if (abs(self.theta - old_theta)).sum() < self.eps:
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
        # *** END CODE HERE ***

    def j(self, x, y):
        n = x.shape[0]
        summation = 0
        for i in range(n):
            summation += y[i] * np.log(self.h(x[i])) + (1 - y[i]) * np.log(1 - self.h(x[i]))
        return -1 / n * summation

    def j_prime(self, x, y):
        n = x.shape[0]
        summation = 0
        for i in range(n):
            summation += y[i] * x[i] - x[i] * self.h(x[i])
        return -1 / n * summation

    def h(self, x):
        return 1 / (1 + np.exp(self.theta.transpose() * x))


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
