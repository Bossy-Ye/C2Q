import unittest
from src.utils import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_matrix_transition1(self):
        matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
        pauli_decomp = decompose_into_pauli(matrix)
        assert np.allclose(pauli_decomp.to_matrix(), matrix), "matrix is not a Pauli decomposition"

    def test_matrix_transition2(self):
        matrix = np.array([[4, 1], [2, 3]], dtype=complex)
        pauli_decomp = decompose_into_pauli(matrix)
        assert np.allclose(pauli_decomp.to_matrix(), matrix), "matrix is not a matrix"

    def test_matrix_transition3(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=complex)
        pauli_decomp = decompose_into_pauli(matrix)
        assert np.allclose(pauli_decomp.to_matrix(), matrix), "matrix is not a matrix"

    def test_num_qubits(self):
        assert qubit_num(2) == 1, "false"
        assert qubit_num(4) == 2, "false"
        assert qubit_num(8) == 3, "false"


if __name__ == '__main__':
    unittest.main()
