import unittest

import matplotlib.pyplot as plt

from classical_to_quantum.applications.graph.Ising import *
from qiskit import qasm2

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_maxcut_init(self):
        maxcut = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G0",
            "MaximumCut")
        try:
            maxcut.plot_results()
        except Exception as e:
            print(e.__str__())
            self.assertEqual(
                'The required operation has not been executed yet. Please execute the necessary steps first.',
                e.__str__())

    def test_maxcut(self):
        maxcut = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G7",
            "MaximumCut")
        maxcut.run()
        maxcut.plot_graph_solution()
        plt.show()

    def test_MIS(self):
        MIS = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G9",
            "MIS")
        MIS.run()
        MIS.plot_graph_solution()
        plt.show()
        circuit, qasm_code = MIS.generate_qasm()
        #circuit = qiskit.qasm3.loads(qasm_code)
        print(circuit)

if __name__ == '__main__':
    unittest.main()
