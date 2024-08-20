import unittest
from classical_to_quantum.applications.arithmetic.quantum_arithmetic import quantum_add
from qiskit.primitives import Sampler


class MyTestCase(unittest.TestCase):
    def test_add(self):
        qc, result = quantum_add(128, 128, 9)
        # sampler = Sampler()
        # result = sampler.run(qc, shots=1024).result()
        print(result)



if __name__ == '__main__':
    unittest.main()
