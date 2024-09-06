import openqaoa.problems.qubo
import qiskit
from openqaoa import create_device
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo

from src.applications.graph.graph_problem import *
from openqaoa.utilities import plot_graph
from openqaoa.problems import *
from openqaoa import QAOA
from qiskit import IBMQ
#from qiskit_optimization.applications.tsp import Tsp
from src.utils import adjacency_matrix_from_adj_dict
from src.applications.graph.ising_auxiliary import *
from src.Framework.interpreter import Interpreter


class_mapping = {
    "Knapsack": Knapsack,
    "SlackFreeKnapsack": SlackFreeKnapsack,
    "MaximumCut": MaximumCut,
    "MinimumVertexCover": MinimumVertexCover,
    "NumberPartition": NumberPartition,
    "ShortestPath": ShortestPath,
    "TSP": TSP,
    "TSP_LP": TSP_LP,
    "PortfolioOptimization": PortfolioOptimization,
    "MIS": MIS,
    "BinPacking": BinPacking,
    "VRP": VRP,
    "SK": SK,
    "BPSP": BPSP,
    "KColor": KColor,
    "FromDocplex2IsingModel": FromDocplex2IsingModel,
    "QUBO": QUBO
}


class Ising(GraphProblem):
    def __init__(self, input_data, class_name, as_real=False):
        self.is_executed = None
        self.qaoa = None
        self.opt_result = None
        self.classical_solution = None
        self.class_name = class_name
        super().__init__(input_data)
        if class_name == 'MinimumVertexCover':
            self.problem = class_mapping[class_name](G=self.graph(), field=1.0, penalty=10)
        elif class_name == 'TSP':
            #G = nx.Graph()
            adj_matrix = adjacency_matrix_from_adj_dict(self.graph().adj)
            #G.add_weighted_edges_from(self.elist)
            self.problem = TSP.random_instance(n_cities=3)
            #self.problem = class_mapping[class_name](distance_matrix=adj_matrix.tolist(), A=2.0, B=1.0)
        elif class_name == 'KColor':
            # 4 colors now
            self.problem = class_mapping[class_name](G=self.graph(), k=4)
        elif class_name == 'VRP':
            pos = get_pos_for_graph(G=self.graph())
            self.problem = class_mapping[class_name](G=self.graph(),
                                                     pos=pos,
                                                     n_vehicles=2)
        else:
            self.problem = class_mapping[class_name](G=self.graph())

        self.qubo = self.problem.qubo

        qaoa = QAOA()
        qaoa.set_circuit_properties(p=3, init_type='ramp')
        # device
        qiskit_device = create_device(location='local', name='qiskit.shot_simulator')
        qaoa.set_device(qiskit_device)
        #
        # # circuit properties
        # qaoa.set_circuit_properties(p=2, param_type='standard', init_type='rand', mixer_hamiltonian='x')
        #
        # # backend properties (already set by default)
        # qaoa.set_backend_properties(prepend_state=None, append_state=None)
        #
        # # classical optimizer properties
        qaoa.set_classical_optimizer(method='nelder-mead', maxiter=200, tol=0.001,
                                     optimization_progress=True, cost_progress=True, parameter_log=True)
        self.qaoa = qaoa

    def plot_graph(self):
        plot_graph(self.graph())

    def plot_results(self):
        if not self.is_executed:
            raise NotExecutedError
        self.opt_result.plot_cost()

    def plot_graph_solution(self):
        solution = self.opt_result.most_probable_states.get('solutions_bitstrings')
        # seperated case for vrp
        if self.class_name == 'VRP':
            solution = self.problem.classical_solution()
            self.problem.plot_solution(solution)
            return
        if self.class_name == 'TSP':
            solutions = self.opt_result.most_probable_states.get('solutions_bitstrings')
            solution = binary_to_tsp_order(solutions[0])
            Interpreter.draw_tsp_solution(self.problem.graph, solution)
            return
        if not self.is_executed:
            raise NotExecutedError

        g = self.graph()
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos, nodelist=[idx for idx, bit in enumerate(solution[0]) if bit == '1'],
                               node_color="tab:red")
        nx.draw_networkx_nodes(g, pos, nodelist=[idx for idx, bit in enumerate(solution[0]) if bit == '0'],
                               node_color="tab:blue")
        nx.draw_networkx_edges(g, pos)

    def run(self, verbose=False):
        self.qaoa.compile(self.qubo)
        self.qaoa.optimize()

        self.is_executed = True
        self.opt_result = self.qaoa.result
        if verbose:
            print(self.opt_result.optimized)
        return self.opt_result

    def generate_qasm(self):
        if not self.is_executed:
            raise NotExecutedError
        variational_params = self.qaoa.optimizer.variational_params
        optimized_angles = self.qaoa.result.optimized['angles']
        variational_params.update_from_raw(optimized_angles)
        optimized_circuit = self.qaoa.backend.qaoa_circuit(variational_params)
        return optimized_circuit, qiskit.qasm2.dumps(optimized_circuit)


def plot_tsp_solution(G: nx.Graph, solution):
    """
    Plots the TSP solution on the graph G.

    Parameters:
    G (nx.Graph): The graph representing the cities and distances.
    solution (list): The list of binary strings representing the TSP solution order.
                     The first node is fixed as the starting city.
    """
    # Convert binary strings to integers
    solution = [int(node, 2) for node in solution]

    pos = nx.spring_layout(G)  # Calculate positions for all nodes
    n = len(solution)

    # Create a directed graph to represent the TSP tour
    G2 = nx.DiGraph()
    G2.add_nodes_from(G.nodes(data=True))

    # Add the edges based on the solution order
    for i in range(n - 1):
        G2.add_edge(solution[i], solution[i + 1], weight=G[solution[i]][solution[i + 1]]['weight'])

    # Connect the last node back to the first to complete the cycle
    G2.add_edge(solution[-1], solution[0], weight=G[solution[-1]][solution[0]]['weight'])

    # Draw the graph with the TSP path highlighted
    plt.figure(figsize=(10, 8))

    # Draw original graph (optional)
    nx.draw_networkx(G, pos, node_color='lightgray', edge_color='lightgray', node_size=500, with_labels=True, alpha=0.6)

    # Draw TSP solution
    nx.draw_networkx_nodes(G2, pos, node_color='red', node_size=600)
    nx.draw_networkx_edges(G2, pos, edgelist=G2.edges, edge_color='blue', width=2)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, font_color='blue')

    plt.title("TSP Solution")
    plt.show()

def binary_to_tsp_order(binary_string):
    """
    Converts a binary string into a list representing the TSP order of cities,
    where the first city (the first '1' in the binary string) is always the starting city.

    Parameters:
    binary_string (str): A binary string representing the selection and order of cities.
                         The string length should be (n-1)^2 where n is the number of cities.

    Returns:
    list: A list of city indices in the order they are visited.
    """
    n = int(len(binary_string) ** 0.5) + 1  # Calculate the number of cities (n)
    order = [0]  # Start with the first city fixed as 0
    # Convert binary string into a list of city indices, starting from the rightmost bit
    for i in range(n - 1):
        city_bits = binary_string[i * (n - 1):(i + 1) * (n - 1)]
        #reversed_city_bits = city_bits[::-1]
        # Determine the city index based on which bit is set to '1'
        for j, bit in enumerate(city_bits):
            if bit == '1':
                city_index = j + 1
                order.append(city_index)
                break
    # Check if order contains exactly n unique numbers
    if len(order) != n or len(set(order)) != n:
        raise ValueError("Invalid TSP order: the order does not contain exactly n unique cities.")
    return order