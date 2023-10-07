class Matrix:

    @staticmethod
    def zeros(n: int, m: int):
        return [[0 for _ in range(m)] for _ in range(n)]

    @staticmethod
    def ones(n: int, m: int):
        return [[1 for _ in range(m)] for _ in range(n)]

    @staticmethod
    def convert_to_matrix(x):
        if x == [] or x == [[]]:
            return [[]]
        if not isinstance(x, list):
            return [[x]]
        else:
            if not isinstance(x[0], list):
                return [x]
            return x

    @staticmethod
    def try_to_convert_from_matrix(x: list):
        if x == [] or x == [[]]:
            return []
        if not isinstance(x[0], list):
            raise ValueError("x must be a matrix!")
        if len(x) == 1:
            x_0 = x[0]
            if len(x_0) > 1:
                return x_0
            else:
                return x_0[0]
        return x

    @staticmethod
    def is_all_elements_greater_than_or_equals(matrix: list, value: float):
        return all([all([el >= value for el in row]) for row in matrix])

    @staticmethod
    def shape(matrix: list):
        if matrix == [[]]:
            return 0, 0
        return len(matrix), len(matrix[0])

    @staticmethod
    def matrices_dimensions_are_equal(fst: list, snd: list):
        fst_shape = Matrix.shape(fst)
        snd_shape = Matrix.shape(snd)
        if fst_shape[0] != snd_shape[0] or fst_shape[1] != snd_shape[1]:
            return False
        return True

    @staticmethod
    def _operation_with_two_matrices(operation: str, fst: list, snd: list):
        if not Matrix.matrices_dimensions_are_equal(fst, snd):
            raise ValueError("The matrices have unequal dimensions!")

        n, m = Matrix.shape(fst)
        match operation:
            case '+': return list([[fst[i][j] + snd[i][j] for j in range(m)] for i in range(n)])
            case '-': return list([[fst[i][j] - snd[i][j] for j in range(m)] for i in range(n)])
            case '*': return list([[fst[i][j] * snd[i][j] for j in range(m)] for i in range(n)])
            case '/': return list([[fst[i][j] / snd[i][j] for j in range(m)] for i in range(n)])

    @staticmethod
    def adding(fst: list, snd: list):
        return Matrix._operation_with_two_matrices('+', fst, snd)

    @staticmethod
    def subtraction(fst: list, snd: list):
        return Matrix._operation_with_two_matrices('-', fst, snd)

    @staticmethod
    def pointwise_multiplication(fst: list, snd: list):
        return Matrix._operation_with_two_matrices('*', fst, snd)

    @staticmethod
    def pointrwise_division(fst: list, snd: list):
        return Matrix._operation_with_two_matrices('/', fst, snd)

    @staticmethod
    def multiplication(fst: list, snd: list):
        n, m = Matrix.shape(fst)
        l, p = Matrix.shape(snd)
        if m != l:
            raise ValueError("Matrices have sizes that are not suitable for multiplication")

        result = Matrix.zeros(n, p)
        for i in range(n):
            for j in range(p):
                scalar_product = 0
                for k in range(m):
                    scalar_product += fst[i][k] * snd[k][j]
                result[i][j] = scalar_product
        return result

    @staticmethod
    def add_vector_to_matrix(matrix: list, vector: list):
        n, m = Matrix.shape(matrix)
        k = len(vector)
        if k != m:
            raise ValueError("Width of the matrix and length of the vector don't match!")

        matrix_from_vector = [[vector[i] for i in range(k)] for _ in range(n)]
        return Matrix.adding(matrix, matrix_from_vector)
