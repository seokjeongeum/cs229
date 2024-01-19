import os

import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train_a, t_train_a = util.load_dataset(
        train_path,
        't',
        True,
    )
    x_test, t_test = util.load_dataset(
        test_path,
        't',
        True,
    )
    clf_a = LogisticRegression()
    clf_a.fit(x_train_a, t_train_a)
    np.savetxt(output_path_true, clf_a.predict(x_test))
    util.plot(x_test, t_test, clf_a.theta, f'{os.path.splitext(output_path_true)[0]}.png')
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train_b, y_train_b = util.load_dataset(
        train_path,
        add_intercept=True,
    )
    clf_b = LogisticRegression()
    clf_b.fit(x_train_b, y_train_b)
    np.savetxt(output_path_naive, clf_b.predict(x_test))
    util.plot(x_test, t_test, clf_b.theta, f'{os.path.splitext(output_path_naive)[0]}.png')
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    x_valid, y_valid = util.load_dataset(
        valid_path,
        add_intercept=True,
    )
    alpha = clf_b.predict(x_valid[y_valid == 1]).mean()
    np.savetxt(output_path_adjusted, clf_b.predict(x_test) / alpha)
    util.plot(
        x_test,
        t_test,
        clf_b.theta,
        f'{os.path.splitext(output_path_adjusted)[0]}.png',
        alpha,
    )
    # *** END CODER HERE


if __name__ == '__main__':
    main(train_path='train.csv',
         valid_path='valid.csv',
         test_path='test.csv',
         save_path='posonly_X_pred.txt')
