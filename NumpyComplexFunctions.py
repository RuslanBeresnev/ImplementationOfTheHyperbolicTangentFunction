import numpy as np

import HyperbolicTangent as ht


def tanh_diff_number(x):
    """
    First differential of the hyperbolic tangent function NumPy implementation. Argument 'x' must be a single number.
    """
    return 1 - np.tanh(x) ** 2


def tanh_diff_matrix(x):
    """
    First differential of the hyperbolic tangent function NumPy implementation. Argument 'x' must be a 2D-matrix.
    """
    n, m = np.shape(x)
    return np.ones((n, m), dtype=int) - np.tanh(x) * np.tanh(x)


def f(x):
    """
    Implementation (via NumPy) of the special f(X) function of the form f(X) = (XW + b), where 'X' is arbitrary 2D-list
    (matrix) with sizes n by m; 'W' is specified 2D-list (matrix) with sizes m by k; and 'b' is list
    (k-dimensional vector)
    """
    return np.tanh(np.dot(np.array(x), np.array(ht.w)) + np.array(ht.b))


def f_diff(x):
    """
    First differential implementation (via NumPy) of the special f(X) function of the form f(X) = (XW + b), where 'X' is
    arbitrary 2D-list (matrix) with sizes n by m; 'W' is specified 2D-list (matrix) with sizes m by k; and 'b' is
    list (k-dimensional vector)
    """
    n, _ = np.shape(x)
    return np.ones((n, ht.w_k), dtype=int) - f(x) * f(x)
