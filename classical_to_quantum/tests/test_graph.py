import unittest
from applications.graph.grover_applications.triangle_finding import TriangleFinding
from classical_to_quantum.applications.graph.ising_applications.tspp import TspP
from classical_to_quantum.applications.graph.ising_applications.max_cut import MaxCut

from classical_to_quantum.applications.graph.ising_applications.vertex_coverp import VertexCoverp

class MyGraphTest(unittest.TestCase):
    def test_something(self):
        tri = TriangleFinding("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7.txt")
        res = tri.search()
        print(res)
        self.assertEqual(True, True)  # add assertion here
    def test_tsp(self):
        tsp = TspP("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/TSPLIB/G1")
        tsp.run(verbose=True)
        tsp.plot_res()
    def test_maxcut(self):
        maxcut = MaxCut("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7.txt")
        maxcut.run(verbose=True)
        maxcut.plot_res()

    def test_vertexcover(self):
        vx = VertexCoverp("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7.txt")
        vx.run(verbose=True)
        vx.plot_res()

if __name__ == '__main__':
    unittest.main()
