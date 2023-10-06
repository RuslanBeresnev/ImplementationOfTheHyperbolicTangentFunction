import multiprocessing as mp


class Matrix:

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
    def identity_matrix(n: int, m: int):
        return [[1] * m] * n

    @staticmethod
    def _operation_with_two_matrices(operation: str, fst: list, snd: list):
        if not Matrix.matrices_dimensions_are_equal(fst, snd):
            raise ValueError("The matrices have unequal dimensions!")
        return list([[eval(f"{el_in_fst} {operation} {el_in_snd}") for el_in_fst, el_in_snd in
                      zip(row_in_fst, row_in_snd)] for row_in_fst, row_in_snd in zip(fst, snd)])

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
    def _clamp(value, lower, upper):
        return lower if value < lower else upper if value > upper else value

    @staticmethod
    def _partial_multiplication(fst: list, snd: list, result: list, start_index: int, end_index: int):
        n, m = Matrix.shape(fst)
        _, k = Matrix.shape(snd)
        for x in range(start_index, end_index):
            for y in range(m):
                for z in range(k):
                    result_x = result[x]
                    result_x[z] += fst[x][y] * snd[y][z]
                    result[x] = result_x
        return result

    @staticmethod
    def multi_process_multiplication(fst: list, snd: list):
        n, m = Matrix.shape(fst)
        l, k = Matrix.shape(snd)
        if m != l:
            raise ValueError("Matrices have sizes that are not suitable for multiplication")

        processes_count = mp.cpu_count()
        rows_count_for_process = n // processes_count + 1
        current_chunk_start_index = 0
        mp_manager = mp.Manager()
        result = mp_manager.list([[0] * k] * n)
        processes = []
        for i in range(processes_count):
            current_chunk_end_index = current_chunk_start_index + rows_count_for_process
            current_chunk_end_index = Matrix._clamp(current_chunk_end_index, 0, n)
            process = mp.Process(target=Matrix._partial_multiplication,
                                 args=(fst, snd, result, current_chunk_start_index, current_chunk_end_index))
            processes.append(process)
            current_chunk_start_index = current_chunk_end_index

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return result

    @staticmethod
    def single_process_multiplication(fst: list, snd: list):
        n, m = Matrix.shape(fst)
        l, k = Matrix.shape(snd)
        if m != l:
            raise ValueError("Matrices have sizes that are not suitable for multiplication")

        result = [[0] * k] * n
        for x in range(n):
            for y in range(m):
                for z in range(k):
                    result[x][z] += fst[x][y] * snd[y][z]
        return result

    @staticmethod
    def add_vector_to_matrix(matrix: list, vector: list):
        n, m = Matrix.shape(matrix)
        k = len(vector)
        if k != m:
            raise ValueError("Width of the matrix and length of the vector don't match!")

        matrix_from_vector = [vector] * n
        return Matrix.adding(matrix, matrix_from_vector)
