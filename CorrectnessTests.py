import unittest
from ddt import ddt, idata
from random import random, randint
import numpy as np

import HyperbolicTangent as ht
import NumpyComplexFunctions as npcf


@ddt
class CorrectnessTests(unittest.TestCase):
    """
    Class with tests checking correctness of functions working
    """

    epsilon = 10 ** -5

    @staticmethod
    def _numbers_enumerator():
        whole_numbers = [randint(0, 6) - 3 for _ in range(10)]
        real_numbers = [random() * 6 - 3 for _ in range(10)]
        numbers_data = whole_numbers + real_numbers
        for element in numbers_data:
            yield element

    @staticmethod
    def _matrices_enumerator():
        whole_numbers_matrices = [[[randint(0, 6) - 3 for _ in range(10)] for _ in range(10)] for _ in
                                  range(10)]
        real_numbers_matrices = [[[random() * 6 - 3 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        matrices_data = whole_numbers_matrices + real_numbers_matrices
        for element in matrices_data:
            yield element

    @staticmethod
    def _enumerator_for_fx_tests():
        whole_numbers_matrices = [[[randint(0, 6) - 3 for _ in range(ht.w_m)] for _ in range(20)] for _ in
                                  range(10)]
        real_numbers_matrices = [[[random() * 6 - 3 for _ in range(ht.w_m)] for _ in range(20)] for _ in range(10)]
        matrices_data = whole_numbers_matrices + real_numbers_matrices
        for element in matrices_data:
            yield element

    @idata(_numbers_enumerator())
    def test_hyperbolic_tangent_for_numbers_calculates_values_correctly(self, x):
        result = ht.tanh_number(x)
        expected = np.tanh(x)
        self.assertAlmostEqual(result, expected, delta=CorrectnessTests.epsilon)

    @idata(_matrices_enumerator())
    def test_hyperbolic_tangent_for_matrices_calculates_values_correctly(self, x):
        result = ht.tanh_matrix(x)
        expected = np.tanh(np.array(x)).tolist()
        self.assertTrue(np.allclose(result, expected, atol=CorrectnessTests.epsilon))

    @idata(_numbers_enumerator())
    def test_hyperbolic_tangent_differential_for_numbers_calculates_values_correctly(self, x):
        result = ht.tanh_diff_number(x)
        expected = ht.tanh_diff_for_number_by_definition(x)
        self.assertAlmostEqual(result, expected, delta=CorrectnessTests.epsilon)

    @idata(_matrices_enumerator())
    def test_hyperbolic_tangent_differential_for_matrices_calculates_values_correctly(self, x):
        result = ht.tanh_diff_matrix(x)
        expected = ht.tanh_diff_for_matrix_by_definition(x)
        self.assertTrue(np.allclose(result, expected, atol=CorrectnessTests.epsilon))

    @idata(_enumerator_for_fx_tests())
    def test_f_function_calculates_values_correctly(self, x):
        result = ht.f(x)
        expected = npcf.f(x)
        self.assertTrue(np.allclose(result, expected, atol=CorrectnessTests.epsilon))

    @idata(_enumerator_for_fx_tests())
    def test_f_diff_function_calculates_values_correctly(self, x):
        result = ht.f_diff(x)
        expected = npcf.f_diff(x)
        self.assertTrue(np.allclose(result, expected, atol=CorrectnessTests.epsilon))


if __name__ == "__main__":
    unittest.main()
