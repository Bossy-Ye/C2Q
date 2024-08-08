from qiskit import qasm3
from classical_to_quantum.applications.graph.gset import *
from qiskit_optimization.applications.tsp import Tsp
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising
from qiskit_optimization.converters import QuadraticProgramToQubo


class TspP(Ising):
    def __init__(self, file_path):
        """
        We need to convert this TSP quadratic program into a QUBO problem,
        as the QP of TSP has constraints that we can replace with penalty terms.
        Constraints: xip (i represents the node and p represents its order)
        \sum{i}(xip) = 1 and \sum{p}(xip) = 1
        """
        #super().__init__(file_path)
        super().__init__(file_path)
        #self.problem = Tsp.parse_tsplib_format(file_path)
        Tsp.parse_tsplib_format()
        self.problem = Tsp.create_random_instance(3, seed = 1323)
        self.qp = self.problem.to_quadratic_program()
        self.qp2qubo = QuadraticProgramToQubo()
        self.qubo = self.qp2qubo.convert(self.qp)
        self.qubitOp, self.offset = self.qubo.to_ising()

    def plot_res(self, transmission=False):
        graph = self.problem.graph
        colors = ["r" for node in graph.nodes]
        pos = [graph.nodes[node]["pos"] for node in graph.nodes]
        order = self.nodes_results
        G2 = nx.DiGraph()
        G2.add_nodes_from(graph)
        n = len(order)
        for i in range(n):
            j = (i + 1) % n
            G2.add_edge(order[i], order[j], weight=graph[order[i]][order[j]]["weight"])
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
        )
        edge_labels = nx.get_edge_attributes(G2, "weight")
        nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)

        if not transmission:
            plt.show()

