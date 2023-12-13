from math import exp
from random import random

# This matrix W and vector b are given to us
w_m = 100
w_k = 1000
w = [[(random() * 2 - 1) / 50 for _ in range(w_k)] for _ in range(w_m)]
b = [(random() * 2 - 1) / 5 for _ in range(w_k)]
# Transpose W for future calculations
w_transposed = [[w[i][j] for i in range(w_m)] for j in range(w_k)]


def shape(matrix: list):
    """
    Get 2D-matrix dimensions -> (height, width)
    """
    if matrix == [[]]:
        return 0, 0
    return len(matrix), len(matrix[0])


def sigmoid_number(x):
    """
    Implementation of the sigmoid function by definition. Argument 'x' must be a single number
    """
    if x > 100:
        return 1
    elif x < -100:
        return 0
    return 1 / (1 + exp(-x))


def sigmoid_matrix(x):
    """
    Implementation of the sigmoid function by definition. Argument 'x' must be a 2D-list (matrix)
    """
    n, m = shape(x)
    return [[sigmoid_number(x[i][j]) for j in range(m)] for i in range(n)]


def tanh_number(x):
    """
    Implementation of the hyperbolic tangent function using sigmoid function. Argument 'x' must be a single number
    """
    return 2 * sigmoid_number(2 * x) - 1


def tanh_matrix(x):
    """
    Implementation of the hyperbolic tangent function using sigmoid function. Argument 'x' must be a 2D-list (matrix)
    """
    n, m = shape(x)
    return [[tanh_number(x[i][j]) for j in range(m)] for i in range(n)]


def tanh_diff_number(x):
    """
    First differential of the hyperbolic tangent function implementation using hyperbolic tangent function.
    Argument 'x' must be a single number
    """
    return 1 - tanh_number(x) ** 2


def tanh_diff_matrix(x):
    """
    First differential of the hyperbolic tangent function implementation using hyperbolic tangent function.
    Argument 'x' must be a 2D-list (matrix)
    """
    n, m = shape(x)
    return [[tanh_diff_number(x[i][j]) for j in range(m)] for i in range(n)]


def tanh_diff_for_number_by_definition(x):
    """
    Find first differential for hyperbolic tangent by definition (via finite differences). 'x' should be a number
    """
    h = 10 ** -10
    return (tanh_number(x + h) - tanh_number(x)) / h


def tanh_diff_for_matrix_by_definition(x):
    """
    Find first differential for hyperbolic tangent by definition (via finite differences). 'x' should be a 2D-matrix.
    """
    n, m = shape(x)
    return [[tanh_diff_for_number_by_definition(x[i][j]) for j in range(m)] for i in range(n)]


def __scalar_product(fst, snd, i, j, count):
    """
    Calculate scalar product of i-th row in fst matrix and j-th row in snd matrix. The length of both rows
    is equal 'count'
    """
    result = 0
    for ind in range(count):
        result += fst[i][ind] * snd[j][ind]
    return result


def f(x):
    """
    Implementation of the special f(X) function of the form f(X) = (XW + b), where 'X' is arbitrary 2D-list (matrix)
    with sizes n by m; 'W' is specified 2D-list (matrix) with sizes m by k; and 'b' is list (k-dimensional vector)
    """
    n, m = shape(x)
    if m != w_m:
        raise ValueError("X and W matrix sizes are not suitable for multiplication!")
    return [[tanh_number(__scalar_product(x, w_transposed, i, j, m) + b[j]) for j in range(w_k)] for i in range(n)]


def f_diff(x):
    """
    First differential implementation of the special f(X) function of the form f(X) = (XW + b), where 'X' is
    arbitrary 2D-list (matrix) with sizes n by m; 'W' is specified 2D-list (matrix) with sizes m by k; and 'b' is
    list (k-dimensional vector)
    """
    n, m = shape(x)
    if m != w_m:
        raise ValueError("X and W matrix sizes are not suitable for multiplication!")
    return [[tanh_diff_number(__scalar_product(x, w_transposed, i, j, m) + b[j])
             for j in range(w_k)] for i in range(n)]
