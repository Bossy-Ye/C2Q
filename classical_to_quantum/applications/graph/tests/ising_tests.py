import unittest

import matplotlib.pyplot as plt
from qiskit import qasm2
from qiskit.primitives import Sampler

from classical_to_quantum.applications.graph.Ising import *


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

    def test_SK(self):
        SK = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G9",
            "SK")
        SK.run()
        SK.plot_results()
        plt.show()
        circuit, str = SK.generate_qasm()
        print(circuit)
        print(str)
        qc = qiskit.qasm2.loads(str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS )
        print(qc)

    def test_vertex_cover(self):
        from qiskit_ibm_provider import IBMProvider

        vc = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G0",
            "MinimumVertexCover")
        res = vc.run()
        print(res.optimized)
        print(res.most_probable_states)
        vc.plot_graph_solution()
        plt.show()
        vc.plot_results()
        plt.show()
        circuit, str = vc.generate_qasm()

        #circuit.measure_all()
        sampler = Sampler()
        result = sampler.run(circuit).result()
        print(result)

    def test_tsp(self):
        tsp = Ising("/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G0",
                    "TSP")
        res = tsp.run()
        print(res)


if __name__ == '__main__':
    unittest.main()
