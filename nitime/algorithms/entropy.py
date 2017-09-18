# -*- coding: utf-8 -*-

import itertools
import numpy as np


def entropy(*X):
    """
    Calculate the entropy of a variable, or joint entropy of several variables.

    Parameters
    ----------
    X : array, or list of arrays
        Variable or variables to compute entropy/joint entropy on

    Notes
    -----
    This function can be used to calculate the entropy of a single variable
    (provided as a single input) or to calculate the joint entropy between two
    variables (provided as a series of inputs)
    """
    n_instances = len(X[0])
    H = 0
    for classes in itertools.product(*[set(x) for x in X]):
        v = np.array([True] * n_instances)
        for predictions, c in zip(X, classes):
            v = np.logical_and(v, predictions == c)
        p = np.mean(v)
        H += -p * np.log2(p) if p > 0 else 0
    return H


def conditional_entropy(x, y):
    """
    The conditional entropy H(X|Y) = H(Y,X) - H(Y). X conditioned on Y
    """
    H_y = entropy(y)
    H_yx = entropy(y, x)
    return H_yx - H_y


def mutual_information(x, y):
    """
    The mutual information between two variables

    MI(X, Y) = H(X) + H(Y) - H(X | Y)

    Parameters
    ----------
    x, y : array

    Returns
    -------
    array : mutual information between x and y
    """
    H_x = entropy(x)
    H_y = entropy(y)
    H_xy = entropy(x, y)
    return H_x + H_y - H_xy


def entropy_cc(x, y):
    """
    The entropy correlation coefficient:

    p(H) = sqrt(MI(X, Y) / 0.5 * (H(X) + H(Y)))
    """
    H_x = entropy(x)
    H_y = entropy(y)
    I_xy = mutual_information(y, x)
    return np.sqrt(I_xy / (0.5 * (H_x + H_y)))


def transfer_entropy(x, y, lag=1):
    """
    Transfer entropy for two given signals.

    Parameters
    ----------
      x : array
        source
      y : array
        target
      lag : int

    Returns
    -------
    array : Transfer entropy from x to y
    """
    # Future of i
    Fi = np.roll(x, -lag)
    # Past of i
    Pi = x
    # Past of j
    Pj = y

    # Transfer entropy
    Inf_from_Pi_to_Fi = conditional_entropy(Fi, Pi)

    # Same as cond_entropy(Fi, Pi_Pj)
    H_y = entropy(Pi, Pj)
    H_yx = entropy(Fi, Pj, Pi)
    Inf_from_Pi_Pj_to_Fi = H_yx - H_y

    TE_from_j_to_i = Inf_from_Pi_to_Fi - Inf_from_Pi_Pj_to_Fi

    return TE_from_j_to_i
