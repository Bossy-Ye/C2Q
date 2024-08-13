import qiskit
from openqaoa import create_device
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals

from classical_to_quantum.applications.graph.graph_problem import *
from openqaoa.utilities import plot_graph
from openqaoa.problems import *
from openqaoa import QAOA

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
    def __init__(self, input_data, class_name):
        self.is_executed = None
        self.qaoa = None
        self.opt_result = None
        self.classical_solution = None

        super().__init__(input_data)
        self.problem = class_mapping[class_name](self.graph())
        self.qubo = self.problem.qubo
        qaoa = QAOA()

        # device
        qiskit_device = create_device(location='local', name='qiskit.statevector_simulator')
        qaoa.set_device(qiskit_device)

        # circuit properties
        qaoa.set_circuit_properties(p=2, param_type='standard', init_type='rand', mixer_hamiltonian='x')

        # backend properties (already set by default)
        qaoa.set_backend_properties(prepend_state=None, append_state=None)

        # classical optimizer properties
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
        if not self.is_executed:
            raise NotExecutedError
        solution = self.opt_result.most_probable_states.get('solutions_bitstrings')
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