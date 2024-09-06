import unittest

from src.algorithms.grover import GroverWrapper
from src.applications.arithmetic.factorization import quantum_factor_mul_oracle
from src.applications.arithmetic.quantum_arithmetic import *
from qiskit.primitives import Sampler


class MyTestCase(unittest.TestCase):
    def test_add(self):
        result, qc = quantum_add(16, 16)
        # sampler = Sampler()
        # result = sampler.run(qc, shots=1024).result()
        self.assertEqual(result, 32)

    def test_mul(self):
        result, qc = quantum_multiplication(11, 11)
        print(result)
        self.assertEqual(result, 121)

    def test_sub(self):
        result, qc = quantum_subtract(-4, 88)
        print(result)
        self.assertEqual(result, -92)
    def test_factor_grover(self):
        oracle, prep_state, obj_bits = quantum_factor_mul_oracle(64)
        grover = GroverWrapper(oracle=oracle,
                               iterations=1,
                               state_preparation=prep_state,
                               objective_qubits=obj_bits
                               )
        grover.run(verbose=True)

if __name__ == '__main__':
    unittest.main()
