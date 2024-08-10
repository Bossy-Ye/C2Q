import unittest

import qiskit

from applications.graph.grover_applications.triangle_finding import TriangleFinding
from classical_to_quantum.applications.graph.ising_applications.tspp import TspP
from classical_to_quantum.applications.graph.ising_applications.max_cut import MaxCut

from classical_to_quantum.applications.graph.ising_applications.vertex_coverp import VertexCoverp


def test_vertexcover():
    vx = VertexCoverp("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7.txt")
    vx.run(verbose=True)
    vx.plot_res()


class MyGraphTest(unittest.TestCase):
    def test_triangle_finding(self):
        tri = TriangleFinding("/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/graph_cases/Gset/G3")
        res = tri.search()
        print(res)
        str = tri.export_to_qasm3()
        print(str)
        tri.plot_res()
        self.assertEqual(True, True)  # add assertion here

    def test_tsp(self):
        tsp = TspP("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/TSPLIB/G1")
        tsp.run(verbose=True)
        print(tsp.generate_qasm())
        tsp.plot_res()

    def test_maxcut(self):
        maxcut = MaxCut("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7.txt")
        maxcut.run(verbose=True)
        maxcut.plot_res()


if __name__ == '__main__':
    unittest.main()
