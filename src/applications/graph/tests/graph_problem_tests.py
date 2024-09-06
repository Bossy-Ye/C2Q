import unittest

from src.applications.graph.graph_problem import GraphProblem
import networkx as nx


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_graph_problem_init_from_file(self):
        graph_problem = GraphProblem(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G0")
        self.assertEqual(graph_problem.num_edges, 5)
        self.assertEqual(graph_problem.num_nodes, 4)
        graph_problem.plot_graph()

    def test_graph_init_from_networkx(self):
        nodes = 5
        edge_probability = 0.8
        g = nx.generators.fast_gnp_random_graph(n=nodes, p=edge_probability, seed=45)
        graph_problem = GraphProblem(g)
        self.assertEqual(graph_problem.num_nodes, 5)
        print(graph_problem.elist)

    def test_graph_init_file_exception(self):
        try:
            graph_problem = GraphProblem(
                "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G0")
        except FileNotFoundError as e:
            self.assertEqual(e, FileNotFoundError)


if __name__ == '__main__':
    unittest.main()
