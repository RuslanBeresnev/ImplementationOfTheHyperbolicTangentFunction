import unittest
from ddt import ddt, data, idata
import random
import math
import numpy as np
import HyperbolicTangent as ht


@ddt
class CorrectnessTests(unittest.TestCase):
    """
    Class with tests checking correctness of functions working
    """

    @staticmethod
    def _common_ienumerator():
        """
        Test data for testing tanh() and tanh_diff() functions
        """
        test_data = [
            5, 0, 7.4, [1, 2, 3], [4.5, 6.71, 8.0, 9.6, 100.56],
            [[2, 4],
             [5, 6]],
            [[random.random() * 10 for _ in range(100)] for _ in range(50)],
            [[random.random() * 10 for _ in range(100)] for _ in range(100)]
        ]
        for element in test_data:
            yield element

    @staticmethod
    def _ienumerator_for_fX_tests():
        """
        Test data for testing f(X) function
        """
        test_data = [
            [[random.random() * 10 for _ in range(10)] for _ in range(10)],
            [[random.random() * 10 for _ in range(10)] for _ in range(10)],
            [[random.random() * 10 for _ in range(10)] for _ in range(10)]
        ]
        for element in test_data:
            yield element

    @staticmethod
    def _data_matches(x, y):
        """
        Check if two values/vectors/matrices have the equal data
        """
        if isinstance(x, list) and isinstance(y, list):
            if isinstance(x[0], list) and isinstance(y[0], list):
                n, m = np.shape(x)
                return all([all([math.fabs(x[i][j] - y[i][j]) < 10 ** -10 for j in range(m)]) for i in range(n)])
            else:
                n = len(x)
                return all([math.fabs(x[i] - y[i]) < 10 ** -10 for i in range(n)])
        else:
            return np.abs(x - y) < 10 ** -10

    @idata(_common_ienumerator())
    def test_hyperbolic_tangent_calculates_values_correctly(self, x):
        """
        Test case for tanh() function
        """
        epsilon = 10 ** -100
        result = ht.HyperbolicTangent.tanh(x, epsilon)
        expected = np.tanh(np.array(x)).tolist()
        self.assertTrue(self._data_matches(result, expected))

    @data(0, 1, 1.24, 0.000124, 100.3434, 10, 2)
    def test_hyperbolic_tangent_differential_calculates_values_correctly(self, x):
        """
        Test case for tanh_diff() function
        """
        epsilon = 10 ** -10
        h = 10 ** -10
        result = ht.HyperbolicTangent.tanh_diff(x, epsilon)
        expected = (ht.HyperbolicTangent.tanh(x + h, epsilon) - ht.HyperbolicTangent.tanh(x, epsilon)) / h
        self.assertAlmostEqual(result, expected, delta=10 ** -5)
