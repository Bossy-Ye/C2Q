import unittest

import matplotlib.pyplot as plt
from qiskit import qasm2
from qiskit.primitives import Sampler

from src.applications.graph.Ising import *
from src.applications.graph.ising_auxiliary import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_maxcut_init(self):
        maxcut = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G0",
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
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G7",
            "MaximumCut")
        maxcut.run()
        maxcut.plot_graph_solution()
        plt.show()

    def test_MIS(self):
        MIS = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G9",
            "MIS")
        MIS.run()
        MIS.plot_graph_solution()
        plt.show()
        circuit, qasm_code = MIS.generate_qasm()
        #circuit = qiskit.qasm3.loads(qasm_code)
        print(circuit)

    def test_SK(self):
        SK = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G9",
            "SK")
        SK.run()
        SK.plot_results()
        plt.show()
        SK.plot_graph_solution()
        plt.show()
        circuit, str = SK.generate_qasm()
        print(circuit)
        print(str)
        qc = qiskit.qasm2.loads(str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        print(qc)

    def test_vertex_cover(self):
        from qiskit_ibm_provider import IBMProvider

        vc = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G3",
            "MinimumVertexCover")
        res = vc.run()
        print(res.optimized)
        print(res.most_probable_states)
        vc.plot_graph_solution()
        plt.show()

    def test_tsp_case1(self):
        tsp = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G4",
            "TSP")
        res = tsp.run(verbose=True)
        solutions = res.most_probable_states.get('solutions_bitstrings')
        print(solutions)
        print(get_tsp_solution(solutions))
    def test_tsp_qaoa_case2(self):
        tsp = TSP.random_instance(n_cities=5)
        plt.show()
        qubo = tsp.qubo
        qaoa = QAOA()
        qaoa.compile(qubo)
        qaoa.optimize(verbose=True)
        res = qaoa.result
        print(res.most_probable_states)
        solutions = res.most_probable_states.get('solutions_bitstrings')
        print(solutions)
        print(get_tsp_solution(solutions))
    def test_4_color(self):
        coloring = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G3",
            "KColor")
        result = coloring.run()
        solutions = result.most_probable_states.get('solutions_bitstrings')
        print(solutions)
        plot_first_valid_coloring_solutions(solutions, coloring)
        plt.show()

    def test_vrp(self):
        # Create a simple graph with 5 nodes and weighted edges
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 10), (0, 2, 15), (0, 3, 20),
                                   (1, 2, 35), (1, 3, 25), (2, 3, 30),
                                   (0, 4, 10), (1, 4, 15), (2, 4, 20),
                                   (3, 4, 25)])
        pos = get_pos_for_graph(G)
        print(G.edges)
        vrp = VRP(G=G, pos=pos, n_vehicles=2)
        qubo = vrp.qubo
        qaoa = QAOA()
        qaoa.compile(qubo)
        qaoa.optimize(verbose=True)
        res = qaoa.result
        solutions = res.most_probable_states.get('solutions_bitstrings')
        print(solutions)
        print(vrp.classical_solution())
        vrp.plot_solution(solution=vrp.classical_solution())
        plt.show()

if __name__ == '__main__':
    unittest.main()
