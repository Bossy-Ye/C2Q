import matplotlib.pyplot as plt
import numpy as np
from sympy.matrices.expressions.factorizations import Factorization

from classical_to_quantum.applications.arithmetic.factorization import quantum_factor_mul_oracle
from classical_to_quantum.applications.graph.grover_applications.graph_oracle import independent_set_to_sat, cnf_to_quantum_oracle
from classical_to_quantum.applications.graph.grover_applications.grover_auxiliary import get_top_measurements, plot_triangle_finding, \
    plot_multiple_graph_colorings, plot_multiple_independent_sets
from classical_to_quantum.applications.graph.ising_auxiliary import plot_first_valid_coloring_solutions
from classical_to_quantum.parser import ProblemParser, ProblemType
from classical_to_quantum.algorithms.vqe_algorithm import VQEAlgorithm
from classical_to_quantum.applications.quantum_machine_learning.quantum_kernel_ml import QMLKernel
from classical_to_quantum.utils import *
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.primitives import Sampler, Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from classical_to_quantum.algorithms.grover import GroverWrapper
import os
import tempfile
from qiskit.circuit.library.phase_oracle import PhaseOracle
from qiskit_aer import Aer
from qiskit import transpile
from openqaoa.problems import *
from classical_to_quantum.applications.graph.grover_applications.graph_color import GraphColor
from classical_to_quantum.applications.graph.graph_problem import GraphProblem
from classical_to_quantum.applications.graph.Ising import Ising
from classical_to_quantum.applications.graph.grover_applications.triangle_finding import TriangleFinding
from classical_to_quantum.applications.arithmetic.quantum_arithmetic import *
from classical_to_quantum.utils import *
from qiskit.visualization import plot_histogram


arithmetic_mapping = {
    "Addition": [quantum_add],
    "Subtraction": [quantum_subtract],
    "Multiplication": [quantum_multiplication]
}
algorithms_mapping = {
    "Knapsack": [Knapsack],
    "SlackFreeKnapsack": [SlackFreeKnapsack],
    "MaximumCut": [Ising],
    "MinimumVertexCover": [Ising],
    "NumberPartition": [NumberPartition],
    "ShortestPath": [Ising],
    "TSP": [Ising],
    "TSP_LP": [TSP_LP],
    "PortfolioOptimization": [PortfolioOptimization],
    "MIS": [Ising, GraphProblem],
    "BinPacking": [BinPacking],
    "VRP": [Ising],
    "SK": [SK],
    "BPSP": [BPSP],
    "KColor": [Ising, GraphColor],
    "FromDocplex2IsingModel": [FromDocplex2IsingModel],
    "QUBO": [QUBO],
    "Triangle": [TriangleFinding]
}


class QASMGenerator:
    def __init__(self):
        self.shots = 1024
        self.observable = None
        self.problem_type = None
        self.parser = None

    def qasm_generate(self, classical_code, verbose=False):
        """

        Parameters
        ----------
        classical_code
        verbose: if True, run on local simulator and print and plot readable results

        Returns
        -------

        """
        self.__init__()
        self.parser = ProblemParser()
        self.parser.parse_code(classical_code)
        self.problem_type = self.parser.problem_type
        qasm_codes = {}
        if verbose:
            print(f'problem type: {self.parser.problem_type} data: {self.parser.data}')
        if self.problem_type == ProblemType.EIGENVALUE:
            self.observable = decompose_into_pauli(self.parser.data)
            algorithm = VQEAlgorithm(self.observable,
                                     qubit_num(len(self.parser.data[0])),
                                     reps=2)
            algorithm.run(verbose=verbose)
            return algorithm.export_to_qasm()
        if self.problem_type == ProblemType.MACHINELEARNING:
            local_vars = {}
            exec(classical_code, {}, local_vars)

            # Variables from the classical code
            X_train = local_vars['X_train']
            y_train = local_vars['y_train']
            X_test = local_vars['X_test']
            y_test = local_vars['y_test']

            qmlk = QMLKernel(locals()['X_train'], locals()['y_train'], locals()['X_test'], locals()['y_test'],
                             model='svc')
            qmlk.run()
            if verbose:
                qmlk.plot_data()
                qmlk.show_result()
            return qmlk.generate_qasm()
        if self.problem_type == ProblemType.CNF:
            dimacs = generate_dimacs(self.parser.data)
            fp = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
            fp.write(dimacs)
            file_name = fp.name
            fp.close()
            oracle = None
            try:
                oracle = PhaseOracle.from_dimacs_file(file_name)
            except ImportError as ex:
                print(ex)
            finally:
                os.remove(file_name)
            wrapper = GroverWrapper(oracle, iterations=2, is_good_state=oracle.evaluate_bitstring)
            wrapper.run(verbose=verbose)
            return {'grover': wrapper.export_to_qasm()}
        if self.problem_type == ProblemType.GRAPH:
            if verbose:
                print(f'-------graph problem type:{self.parser.specific_graph_problem}--------')
            algorithms = algorithms_mapping.get(self.parser.specific_graph_problem)
            for algorithm in algorithms:
                if verbose: print(algorithm)
                if issubclass(algorithm, Ising):
                    problem = Ising(self.parser.data, self.parser.specific_graph_problem)
                    res = problem.run(verbose=verbose)
                    if verbose:
                        if self.parser.specific_graph_problem == 'KColor':
                            solutions = res.most_probable_states.get('solutions_bitstrings')
                            print(solutions)
                            plot_first_valid_coloring_solutions(solutions, problem)
                        else:
                            problem.plot_graph_solution()
                            plt.show()
                    _, qasm_codes['qaoa'] = problem.generate_qasm()
                elif issubclass(algorithm, TriangleFinding):
                    problem = TriangleFinding(self.parser.data)
                    res = problem.run(verbose=verbose)
                    if verbose:
                        top_measurements = get_top_measurements(res, 0.001, num=20)
                        plot_triangle_finding(problem.graph(), top_measurements)
                    qasm_codes['grover'] = problem.export_to_qasm()
                elif issubclass(algorithm, GraphColor):
                    problem = GraphColor(self.parser.data)
                    res = problem.run(verbose=verbose)
                    if verbose:
                        top_measurements = get_top_measurements(res, 0.001, num=20)
                        plot_multiple_graph_colorings(problem.graph(), top_measurements, num_per_row=3)
                    qasm_codes['grover'] = problem.export_to_qasm()
                elif issubclass(algorithm, GraphProblem):
                    problem = GraphProblem(self.parser.data)
                    independent_set_cnf = independent_set_to_sat(problem.graph())
                    independent_set_oracle = cnf_to_quantum_oracle(independent_set_cnf)
                    grover = GroverWrapper(oracle=independent_set_oracle,
                                           iterations=2,
                                           objective_qubits=list(range(problem.num_nodes)))
                    res = grover.run(verbose=verbose)
                    qasm_codes['grover'] = grover.export_to_qasm()
                    if verbose:
                        if self.parser.specific_graph_problem == 'MIS':
                            top_is_measurements = get_top_measurements(res, num=100)
                            plot_multiple_independent_sets(problem.graph(), top_is_measurements)

            return qasm_codes
        elif self.problem_type == ProblemType.ARITHMETICS:
            left = self.parser.data.get('left')
            right = self.parser.data.get('right')
            operation = arithmetic_mapping.get(self.parser.specific_arithmetic_operation)
            res, circuit = operation[0](left, right)
            if verbose: print(f'quantum {self.parser.specific_arithmetic_operation} result: {res}')
            qasm_codes['QFT'] = qasm2.dumps(circuit)
            return qasm_codes
        elif self.problem_type == ProblemType.FACTOR:
            number = self.parser.data.get('composite number')
            oracle, prep_state, obj_bits = quantum_factor_mul_oracle(number)
            grover = GroverWrapper(oracle=oracle,
                                   iterations=1,
                                   state_preparation=prep_state,
                                   objective_qubits=obj_bits
                                   )
            res = grover.run(verbose=False)
            if verbose:
                plot_histogram(res.circuit_results)
                plt.show()
            solutions = get_top_measurements(res)
            if verbose: print(solutions)
            qasm_codes['grover'] = grover.export_to_qasm()
            return qasm_codes
        else:
            raise ValueError("Unsupported problem type")

    def run_locally(self):
        pass

    def run_qasm_simulator(self, str, primitive: str = 'sampler'):
        seed = int(np.random.randint(1, 1000000))
        circuit = qasm2.loads(str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        pm = generate_preset_pass_manager(optimization_level=1)
        optimized_circuit = pm.run(circuit)
        if primitive == "sampler":
            #optimized_circuit.measure_all()
            sampler = Sampler()
            result = sampler.run(optimized_circuit, seed=seed).result()
            return result
        elif primitive == "estimator":
            isa_observable = self.observable.apply_layout(optimized_circuit.layout)
            estimator = Estimator()
            result = estimator.run(optimized_circuit, isa_observable, shots=2048, seed=seed).result()
            return result

    def run_qasm_aer(self, str, primitive: str = 'sampler', noise=None):
        circuit = qasm2.loads(str)
        circuit.measure_all()
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=self.shots).result()
        return result.get_counts(compiled_circuit)

# Example usage
# cnf_formula = [[1, -2], [2, 3, -1]]
# dimacs_string = generate_dimacs(cnf_formula)
# print(dimacs_string)
