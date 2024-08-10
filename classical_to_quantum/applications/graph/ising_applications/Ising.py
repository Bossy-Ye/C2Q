from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals

from classical_to_quantum.applications.graph.graph_problem import *
from openqaoa.utilities import plot_graph


class Ising(GraphProblem):
    def __init__(self, filepath):
        self.result = None
        self.solution = None
        self.is_executed_search = None
        self.state_vector = None
        self.ansatz = None
        self.qp = None
        self.qubitOp = None
        self.converge_vals = None
        self.converge_counts = None
        self.problem = None
        self.results = []
        self.reps_results = []
        self.is_executed = False
        self.opt_solver = OptSolver()
        super().__init__(file_path=filepath)

    def plot_graph(self):
        plot_graph(self.graph())

    def run(self, verbose=False):
        if self.is_executed:
            print("already executed")
            return
        algorithm_globals.random_seed = 1111
        self.ansatz = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks="cz",
                               reps=2, entanglement="linear", num_qubits=self.qubitOp.num_qubits)
        log = OptimizerLog()
        sampler = Sampler()
        veq = SamplingVQE(sampler=sampler, optimizer=COBYLA(),
                          callback=log.update,
                          ansatz=self.ansatz)
        result = veq.compute_minimum_eigenvalue(self.qubitOp)
        self.result = result
        self.is_executed = True
        self.state_vector = result.eigenstate
        self.nodes_results = self.interpret(self.problem.sample_most_likely(self.state_vector))
        if verbose:
            print(result)
        if verbose:
            print(f'eigenvalue {result.eigenvalue}')
            print(f'vertices {self.nodes_results}')
        return result, log

    def run_search_parameters(self, verbose=False, CVaR=False):
        if self.is_executed_search:
            return
        optimizers = self.opt_solver.optimizers
        self.converge_counts = np.empty([len(optimizers)], dtype=object)
        self.converge_vals = np.empty([len(optimizers)], dtype=object)
        if CVaR:
            alphas = [1.0, 0.50, 0.25]
        else:
            alphas = [1.0]
        for i, optimizer in enumerate(optimizers):
            optimizer_name = type(optimizer).__name__
            all_counts = []
            all_values = []
            for alpha in alphas:
                if verbose:
                    print("\rOptimizer: {} Alpha: {}       ".format(optimizer_name, alpha), end="")
                algorithm_globals.random_seed = 1234
                ansatz = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks="cz",
                                  reps=2, entanglement="linear", num_qubits=self.qubitOp.num_qubits)
                log = OptimizerLog()
                sampler = Sampler()
                veq = SamplingVQE(sampler=sampler, optimizer=optimizer,
                                  callback=log.update, aggregation=alpha,
                                  ansatz=ansatz)

                result = veq.compute_minimum_eigenvalue(self.qubitOp)
                #self.result = result
                if verbose:
                    print(f'eigenvalue {result.eigenvalue} alpha {alpha}')

                all_counts.append(np.asarray(log.get_counts()))
                all_values.append(np.asarray(log.get_values()))
                self.results.append((optimizer_name, alpha, result))

            self.converge_counts[i] = all_counts
            self.converge_vals[i] = all_values
        if verbose:
            print("\rOptimization complete      ")

        self.is_executed_search = True

    def show_results(self, plot=True):
        if not self.is_executed:
            print("Not executed")
            return
        else:
            print(f"Quantum computer gives solution: {self.problem.sample_most_likely(self.state_vector)}")

    def show_search_results(self):
        if not self.is_executed_search:
            print("Search not executed")
            return
        else:
            for optimizer_name, alpha, result in self.results:
                print(
                    f"{optimizer_name} (Alpha {alpha}) gave Solution: {self.problem.sample_most_likely(result.eigenstate)}")
            solver = NumPyMinimumEigensolver()
            result = solver.compute_minimum_eigenvalue(self.qubitOp)
            print(f"Classical result (eigenvalue): {result.eigenvalue.real}")
            print(f"Classical solution: {self.problem.sample_most_likely(result.eigenstate)}")

    def layer_wise_run(self, max_num_layers=5, verbose=False):
        optimal = []
        for reps in range(max_num_layers):
            circuit = TwoLocal(rotation_blocks=['ry', 'rz'],
                               entanglement_blocks="cz",
                               entanglement="linear",
                               num_qubits=self.qubitOp.num_qubits,
                               reps=reps + 1)
            log = OptimizerLog()
            sampler = Sampler()
            veq = SamplingVQE(sampler=sampler, optimizer=COBYLA(),
                              callback=log.update, ansatz=circuit)
            result = veq.compute_minimum_eigenvalue(self.qubitOp)
            optimal.append(result)
            self.reps_results.append((reps + 1, circuit, result))
            if verbose:
                print('Layer:', reps + 1, ' Best Value:', result.eigenvalue)
        return optimal

    def generate_qasm(self):
        if not self.is_executed:
            raise ValueError("Not executed")

        optimized_params = self.result.optimal_point
        bound_ansatz = self.ansatz.assign_parameters(optimized_params)

        # Convert the bound ansatz circuit to QASM format
        qasm_code = qasm2.dumps(bound_ansatz)
        return qasm_code

    def plot_search_res(self):
        pylab.rcParams["figure.figsize"] = (12, 8)
        for i, optimizer in enumerate(self.opt_solver.optimizers):
            optimizer_name = type(optimizer).__name__
            for j, alpha in enumerate([1.0, 0.50, 0.25] if len(self.converge_counts[i]) > 1 else [1.0]):
                pylab.plot(self.converge_counts[i][j], self.converge_vals[i][j].real,
                           label=f"{optimizer_name} α={alpha}")
        pylab.xlabel("Eval count")
        pylab.ylabel("Energy")
        pylab.title("Energy convergence for various optimizers")
        pylab.legend(loc="upper right")
        pylab.show()

    def interpret(self, x):
        if not self.is_executed:
            raise ValueError("Not executed")
        return self.problem.interpret(x)
