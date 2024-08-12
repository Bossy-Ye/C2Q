import unittest
from quantum_arithmetic import quantum_add
from qiskit.primitives import Sampler


class MyTestCase(unittest.TestCase):
    def test_add(self):
        qc = quantum_add(128, 128, 8)
        sampler = Sampler()
        result = sampler.run(qc, shots=1024).result()
        self.assertEqual(result.quasi_dists[0], {256: 1.0})  # add assertion here


if __name__ == '__main__':
    unittest.main()
