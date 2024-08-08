import numpy as np
import qiskit.qasm3

from classical_to_quantum.parser import ProblemParser, ProblemType
from classical_to_quantum.algorithms.vqe_algorithm import VQEAlgorithm
from classical_to_quantum.applications.quantum_machine_learning.quantum_kernel_ml import QMLKernel
from classical_to_quantum.utils import *
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler, Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from classical_to_quantum.algorithms.grover import GroverWrapper
import os
import tempfile
from qiskit import transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit.library.phase_oracle import PhaseOracle
from qiskit_aer import Aer
from qiskit import transpile


def generate_dimacs(cnf_formula):
    num_vars = max(abs(var) for clause in cnf_formula for var in clause)
    num_clauses = len(cnf_formula)

    # Start with the problem line
    dimacs_str = f"p cnf {num_vars} {num_clauses}\n"

    # Add each clause
    for clause in cnf_formula:
        dimacs_str += ' '.join(map(str, clause)) + ' 0\n'

    return dimacs_str


class QASMGenerator:
    def __init__(self):
        self.shots = 1024
        self.observable = None
        self.problem_type = None
        self.parser = None

    def qasm_generate(self, classical_code, verbose=False):
        self.parser = ProblemParser()
        self.problem_type = self.parser.parse_code(classical_code)
        if verbose:
            print(self.parser.variables)
        if self.problem_type == ProblemType.EIGENVALUE:
            self.observable = decompose_into_pauli(self.parser.observable)
            algorithm = VQEAlgorithm(self.observable,
                                     qubit_num(len(self.parser.observable[0])),
                                     reps=2)
            algorithm.generate_quantum_code()
            return algorithm.export_to_qasm3()
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
            return qmlk.generate_qasm3()
        if self.problem_type == ProblemType.CNF:
            dimacs = generate_dimacs(self.parser.variables.get("cnf_formula"))
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
            wrapper = GroverWrapper(oracle)
            wrapper.generate_quantum_code(verbose=True)
            return wrapper.export_to_qasm3()
        else:
            raise ValueError("Unsupported problem type")

    def run_qasm(self, str, primitive: str = 'sampler'):
        seed = int(np.random.randint(1, 1000000))
        circuit = qiskit.qasm3.loads(str)
        pm = generate_preset_pass_manager(optimization_level=1)
        optimized_circuit = pm.run(circuit)
        if primitive == "sampler":
            optimized_circuit.measure_all()
            sampler = Sampler()
            result = sampler.run(optimized_circuit, seed=seed).result()
            return result
        elif primitive == "estimator":
            isa_observable = self.observable.apply_layout(optimized_circuit.layout)
            estimator = Estimator()
            result = estimator.run(optimized_circuit, isa_observable, shots=2048, seed=seed).result()
            return result

    def run_qasm_aer(self, str, primitive: str = 'sampler', noise=None):
        circuit = qiskit.qasm3.loads(str)
        circuit.measure_all()
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=self.shots).result()
        return result.get_counts(compiled_circuit)

# Example usage
# cnf_formula = [[1, -2], [2, 3, -1]]
# dimacs_string = generate_dimacs(cnf_formula)
# print(dimacs_string)



