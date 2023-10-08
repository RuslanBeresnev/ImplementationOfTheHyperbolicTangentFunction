from Matrix import Matrix


class HyperbolicTangent:
    # This matrix and vector are given to us
    _w = [[(i + 1) * (j + 1) for j in range(10)] for i in range(10)]
    _b = [i for i in range(10)]

    @staticmethod
    def _exp(x: list, epsilon: float):
        shape = Matrix.shape(x)
        matrix_of_ones = Matrix.ones(shape[0], shape[1])

        factorial_multiplier = matrix_of_ones
        term_factor = Matrix.pointrwise_division(x, factorial_multiplier)
        term = matrix_of_ones
        maclaurin_row = term

        while Matrix.is_all_elements_greater_than_or_equals(term, epsilon):
            term = Matrix.pointwise_multiplication(term, term_factor)
            maclaurin_row = Matrix.adding(maclaurin_row, term)
            factorial_multiplier = Matrix.adding(factorial_multiplier, matrix_of_ones)
            term_factor = Matrix.pointrwise_division(x, factorial_multiplier)

        return maclaurin_row

    @staticmethod
    def tanh(x, epsilon: float):
        if epsilon == 0:
            raise ValueError("Epsilon can't take the value zero!")
        if x == [] or x == [[]]:
            return []

        x = Matrix.convert_to_matrix(x)
        shape = Matrix.shape(x)
        matrix_of_ones = Matrix.ones(shape[0], shape[1])

        exp_x = HyperbolicTangent._exp(x, epsilon)
        reciprocal_exp_x = Matrix.pointrwise_division(matrix_of_ones, exp_x)
        result = Matrix.pointrwise_division(Matrix.subtraction(exp_x, reciprocal_exp_x),
                                            Matrix.adding(exp_x, reciprocal_exp_x))
        result = Matrix.try_to_convert_from_matrix(result)
        return result

    @staticmethod
    def tanh_diff(x, epsilon: float):
        if epsilon == 0:
            raise ValueError("Epsilon can't take the value zero!")
        if x == [] or x == [[]]:
            return []

        x = Matrix.convert_to_matrix(x)
        shape = Matrix.shape(x)
        matrix_of_ones = Matrix.ones(shape[0], shape[1])

        tanh_x = HyperbolicTangent.tanh(x, epsilon)
        tanh_x = Matrix.convert_to_matrix(tanh_x)
        result = Matrix.subtraction(matrix_of_ones, Matrix.pointwise_multiplication(tanh_x, tanh_x))
        result = Matrix.try_to_convert_from_matrix(result)
        return result

    @staticmethod
    def f(x: list, epsilon: float):
        x = Matrix.convert_to_matrix(x)
        x_width = Matrix.shape(x)[1]
        w_height = Matrix.shape(HyperbolicTangent._w)[0]
        if x_width != w_height:
            raise ValueError("Width of X matrix doesn't equal height of W matrix!")

        xw = Matrix.multiplication(x, HyperbolicTangent._w)
        xw_plus_b = Matrix.add_vector_to_matrix(xw, HyperbolicTangent._b)
        result = HyperbolicTangent.tanh(xw_plus_b, epsilon)
        result = Matrix.try_to_convert_from_matrix(result)
        return result
