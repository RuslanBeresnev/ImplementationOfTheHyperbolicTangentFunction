import numpy as np
import HyperbolicTangent as ht
import random
import time


class OperatingTimeComparison:
    """
    Class for measuring the speed of functions and comparing them
    """

    @staticmethod
    def _make_launches(x, epsilon: float, launches_count: int):
        """
        Perform launches to calculate operating time of the functions
        """

        start_time = time.time()
        for i in range(launches_count):
            ht.HyperbolicTangent.tanh(x, epsilon)
        end_time = time.time()
        implemented_tanh_average_operating_time = (end_time - start_time) / launches_count

        start_time = time.time()
        for i in range(launches_count):
            np.tanh(x)
        end_time = time.time()
        numpy_tanh_average_operating_time = (end_time - start_time) / launches_count

        start_time = time.time()
        for i in range(launches_count):
            ht.HyperbolicTangent.tanh_diff(x, epsilon)
        end_time = time.time()
        implemented_tanh_diff_average_operating_time = (end_time - start_time) / launches_count

        start_time = time.time()
        for i in range(launches_count):
            1 - np.tanh(x) ** 2
        end_time = time.time()
        numpy_tanh_diff_average_operating_time = (end_time - start_time) / launches_count

        return [(implemented_tanh_average_operating_time, numpy_tanh_average_operating_time),
                (implemented_tanh_diff_average_operating_time, numpy_tanh_diff_average_operating_time)]

    @staticmethod
    def time_measurements_for_number(launches_count: int):
        """
        Calculate operating time for number
        """
        x = 0.65748
        epsilon = 10 ** -10
        return OperatingTimeComparison._make_launches(x, epsilon, launches_count)

    @staticmethod
    def time_measurements_for_matrix(launches_count: int):
        """
        Calculate operating time for matrix
        """
        x = [[random.random() * 10 for _ in range(100)] for _ in range(100)]
        epsilon = 10 ** -10
        return OperatingTimeComparison._make_launches(x, epsilon, launches_count)

    @staticmethod
    def print_result_table():
        """
        Show all results
        """
        results_for_number = OperatingTimeComparison.time_measurements_for_number(100)
        results_for_matrix = OperatingTimeComparison.time_measurements_for_matrix(100)

        print()
        print()

        print("Approximate results for number (in milliseconds):")
        print("-" * 50)
        print("\t\t\t tanh() \t tanh_differential()")
        print(f"Manual: \t {round(results_for_number[0][0] * 1000, 2)} \t\t {round(results_for_number[1][0] * 1000, 2)}")
        print(f"NumPy: \t\t {round(results_for_number[0][1] * 1000, 2)} \t\t {round(results_for_number[1][1] * 1000, 2)}")
        print("-" * 50)
        tanh_boost = round(results_for_number[0][0] / results_for_number[0][1], 1)
        print(f"The tanh() function in the NumPy module approximately is {tanh_boost} times faster than the manual "
              f"implementation")
        tanh_diff_boost = round(results_for_number[1][0] / results_for_number[1][1], 1)
        print(f"The tanh_differential() function in the NumPy module approximately is {tanh_diff_boost} times faster "
              f"than the manual implementation")

        print()
        print()
        print()

        print("Approximate results for matrix (in milliseconds):")
        print("-" * 50)
        print("\t\t\t tanh() \t tanh_differential()")
        print(f"Manual: \t {round(results_for_matrix[0][0] * 1000, 2)} \t\t {round(results_for_matrix[1][0] * 1000, 2)}")
        print(f"NumPy: \t\t {round(results_for_matrix[0][1] * 1000, 2)} \t\t {round(results_for_matrix[1][1] * 1000, 2)}")
        print("-" * 50)
        tanh_boost = round(results_for_matrix[0][0] / results_for_matrix[0][1], 1)
        print(f"The tanh() function in the NumPy module approximately is {tanh_boost} times faster than the manual "
              f"implementation")
        tanh_diff_boost = round(results_for_matrix[1][0] / results_for_matrix[1][1], 1)
        print(f"The tanh_differential() function in the NumPy module approximately is {tanh_diff_boost} times faster "
              f"than the manual implementation")


if __name__ == '__main__':
    OperatingTimeComparison.print_result_table()
