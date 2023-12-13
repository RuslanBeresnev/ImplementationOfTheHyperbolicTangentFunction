from random import random
from time import time
import numpy as np
from prettytable import PrettyTable

import HyperbolicTangent as ht
import NumpyComplexFunctions as npcf

LAUNCHES_COUNT_FOR_NUMBER = 1000
LAUNCHES_COUNT_FOR_MATRIX = 10


def _make_launches_for_function_and_data(func, x):
    """
    Perform launches to calculate average operating time (in ms) of the function applied to data
    """
    if not isinstance(x, list):
        launches_count = LAUNCHES_COUNT_FOR_NUMBER
        x_is_number = True
    else:
        launches_count = LAUNCHES_COUNT_FOR_MATRIX
        x_is_number = False

    start_time = time()
    for _ in range(launches_count):
        func(x)
    end_time = time()

    result = (end_time - start_time) * 1000
    # If 'x' is a number, then we calculate the TOTAL TIME for all runs,
    # since the execution time of one operation is too short
    if not x_is_number:
        result /= launches_count
    return round(result, 2)


def _time_measurements_for_number(number):
    """
    Calculate operating time of the tanh(), tanh_diff() functions and their NumPy implementations for the number
    """
    manual_tanh = _make_launches_for_function_and_data(ht.tanh_number, number)
    manual_tanh_diff = _make_launches_for_function_and_data(ht.tanh_diff_number, number)

    numpy_tanh = _make_launches_for_function_and_data(np.tanh, number)
    numpy_tanh_diff = _make_launches_for_function_and_data(npcf.tanh_diff_number, number)

    difference_for_tanh = round(manual_tanh / numpy_tanh, 1)
    difference_for_tanh_diff = round(manual_tanh_diff / numpy_tanh_diff, 1)

    results = {
        'manual_tanh': manual_tanh,
        'manual_tanh_diff': manual_tanh_diff,
        'numpy_tanh': numpy_tanh,
        'numpy_tanh_diff': numpy_tanh_diff,
        'difference_for_tanh': difference_for_tanh,
        'difference_for_tanh_diff': difference_for_tanh_diff
    }
    return results


def _time_measurements_for_matrix(matrix):
    """
    Calculate operating time of the tanh(), tanh_diff(), f(), f_diff() functions and their NumPy implementations
    for the 2D-matrix
    """
    manual_tanh = _make_launches_for_function_and_data(ht.tanh_matrix, matrix)
    manual_tanh_diff = _make_launches_for_function_and_data(ht.tanh_diff_matrix, matrix)
    manual_f = _make_launches_for_function_and_data(ht.f, matrix)
    manual_f_diff = _make_launches_for_function_and_data(ht.f_diff, matrix)

    numpy_tanh = _make_launches_for_function_and_data(np.tanh, matrix)
    numpy_tanh_diff = _make_launches_for_function_and_data(npcf.tanh_diff_matrix, matrix)
    numpy_f = _make_launches_for_function_and_data(npcf.f, matrix)
    numpy_f_diff = _make_launches_for_function_and_data(npcf.f_diff, matrix)

    difference_for_tanh = round(manual_tanh / numpy_tanh, 1)
    difference_for_tanh_diff = round(manual_tanh_diff / numpy_tanh_diff, 1)
    difference_for_f = round(manual_f / numpy_f, 1)
    difference_for_f_diff = round(manual_f_diff / numpy_f_diff, 1)

    results = {
        'manual_tanh': manual_tanh,
        'manual_tanh_diff': manual_tanh_diff,
        'manual_f': manual_f,
        'manual_f_diff': manual_f_diff,
        'numpy_tanh': numpy_tanh,
        'numpy_tanh_diff': numpy_tanh_diff,
        'numpy_f': numpy_f,
        'numpy_f_diff': numpy_f_diff,
        'difference_for_tanh': difference_for_tanh,
        'difference_for_tanh_diff': difference_for_tanh_diff,
        'difference_for_f': difference_for_f,
        'difference_for_f_diff': difference_for_f_diff
    }
    return results


def print_result_tables():
    """
    Show all results for operating time
    """
    number = 0.78397
    n, m = 50, ht.w_m
    matrix = [[random() * 6 - 3 for _ in range(m)] for _ in range(n)]

    results_for_number = _time_measurements_for_number(number)
    results_for_matrix = _time_measurements_for_matrix(matrix)

    table_for_number = PrettyTable()
    table_for_number.field_names = ['Implementation\\Function', 'tanh()', 'tanh_diff()']
    table_for_number.add_row(['Manual', results_for_number['manual_tanh'], results_for_number['manual_tanh_diff']])
    table_for_number.add_row(['NumPy', results_for_number['numpy_tanh'], results_for_number['numpy_tanh_diff']])
    table_for_number.add_row(['NumPy is n times faster', results_for_number['difference_for_tanh'],
                              results_for_number['difference_for_tanh_diff']])

    table_for_matrix = PrettyTable()
    table_for_matrix.field_names = ['Implementation\\Function', 'tanh()', 'tanh_diff()', 'f()', 'f_diff()']
    table_for_matrix.add_row(['Manual', results_for_matrix['manual_tanh'], results_for_matrix['manual_tanh_diff'],
                              results_for_matrix['manual_f'], results_for_matrix['manual_f_diff']])
    table_for_matrix.add_row(['NumPy', results_for_matrix['numpy_tanh'], results_for_matrix['numpy_tanh_diff'],
                              results_for_matrix['numpy_f'], results_for_matrix['numpy_f_diff']])
    table_for_matrix.add_row(['NumPy is n times faster', results_for_matrix['difference_for_tanh'],
                              results_for_matrix['difference_for_tanh_diff'], results_for_matrix['difference_for_f'],
                              results_for_matrix['difference_for_f_diff']])

    print(f'\nApproximate results of calculation performance (TOTAL TIME IN MILLISECONDS FOR '
          f'{LAUNCHES_COUNT_FOR_NUMBER} LAUNCHES) for the number {number}, and comparison '
          f'of the speed of implementation via NumPy and manual implementation: ')
    print(table_for_number)

    print(f'\nApproximate results of calculation performance (TIME IN MILLISECONDS FOR ONE LAUNCH) for the {n} by {m} '
          f'matrix and comparison of the speed of implementation via NumPy and manual '
          f'implementation: ')
    print(table_for_matrix)


if __name__ == '__main__':
    print_result_tables()
