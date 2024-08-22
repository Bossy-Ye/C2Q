import qiskit
from openqaoa import create_device
from classical_to_quantum.applications.graph.graph_problem import *
from openqaoa.utilities import plot_graph
from openqaoa.problems import *
from openqaoa import QAOA
from qiskit import IBMQ
from classical_to_quantum.utils import adjacency_matrix_from_adj_dict
from classical_to_quantum.applications.graph.ising_auxiliary import *
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
            self.problem = class_mapping[class_name](distance_matrix=adj_matrix.tolist(), A=2.0, B=1.0)
        elif class_name == 'KColor':
            # 4 colors now
            self.problem = class_mapping[class_name](G=self.graph(), k=4)
        elif class_name == 'VRP':
            print(self.graph().edges)
            pos = get_pos_for_graph(G=self.graph())
            self.problem = class_mapping[class_name](G=self.graph(),
                                                     pos = pos,
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
        # seperated case for vrp
        if self.class_name == 'VRP':
            solution = self.problem.classical_solution()
            self.problem.plot_solution(solution)
            return
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


