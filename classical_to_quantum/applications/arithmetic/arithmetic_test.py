import unittest
from classical_to_quantum.applications.arithmetic.quantum_arithmetic import *
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

if __name__ == '__main__':
    unittest.main()
