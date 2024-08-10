# tests/test_vqe_algorithm.py
import unittest
from qiskit.quantum_info import SparsePauliOp
from classical_to_quantum.algorithms.vqd_algorithm import VQDAlgorithm


class TestVQDAlgorithm(unittest.TestCase):
    def test_generate_quantum_code(self):
        observable = SparsePauliOp.from_list([("II", 2), ("XX", -2),
                                              ("YY", 3), ("ZZ", -3)])
        vqe = VQDAlgorithm(observable, 2, n_eigenvectors=3,
                           run_as_classical=True)
        vqe.run()
        print(vqe.export_to_qasm())


if __name__ == '__main__':
    unittest.main()
