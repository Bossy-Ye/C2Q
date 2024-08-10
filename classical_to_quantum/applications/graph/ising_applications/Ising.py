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
    def __init__(self, filepath, class_name):
        self.result = None
        self.is_executed = None
        self.qaoa = None
        super().__init__(file_path=filepath)
        self.problem = class_mapping[class_name]()
        self.maxcut_qubo = self.problem.qubo
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

    def run(self, verbose=False):
        self.qaoa.compile(self.maxcut_qubo)
        self.qaoa.optimize()

        self.is_executed = True
        result = self.qaoa.result
        if verbose:
            print(result.optimized)
        return result

    def generate_qasm(self):
        if not self.is_executed:
            raise ValueError("not executed yet")
        variational_params = self.qaoa.optimizer.variational_params
        optimized_angles = self.qaoa.result.optimized['angles']
        variational_params.update_from_raw(optimized_angles)
        optimized_circuit = self.qaoa.backend.qaoa_circuit(variational_params)
        return qiskit.qasm2.dumps(optimized_circuit)
