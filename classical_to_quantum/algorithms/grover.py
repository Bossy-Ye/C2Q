from qiskit.circuit import QuantumCircuit

# input_3sat_instance = """
# p cnf 3 5
# -1 -2 -3 0
# 1 -2 3 0
# 1 2 -3 0
# 1 -2 -3 0
# -1 2 3 0
# """

from qiskit_algorithms import AmplificationProblem

from classical_to_quantum.algorithms.base_algorithm import BaseAlgorithm
from qiskit.circuit.library import PhaseOracle, GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit import qasm3
from qiskit.primitives import Sampler


class GroverWrapper(BaseAlgorithm):
    def __init__(self,
                 oracle: QuantumCircuit,
                 iteration=None,
                 state_preparation: QuantumCircuit = None,
                 is_good_state=None,
                 objective_qubits=None
                 ):
        super().__init__()
        self.grover = None
        if isinstance(oracle, PhaseOracle):
            is_good_state = oracle.evaluate_bitstring
        self.operator = None
        self.oracle = oracle
        self.iteration = iteration

        if oracle is not None:
            self.problem = AmplificationProblem(oracle,
                                                state_preparation=state_preparation,
                                                is_good_state=is_good_state,
                                                objective_qubits=objective_qubits)

    def run(self, verbose=False):
        self.grover = Grover(iterations=self.iteration, sampler=Sampler())
        result = self.grover.amplify(self.problem)
        if verbose:
            print(result)
        return result

    def run_on_quantum(self):
        return None

    def export_to_qasm(self):
        if self.operator is None:
            raise ValueError("Grover operator has not been generated yet. Call generate_quantum_code() first.")
        qasm_str = qasm3.dumps(self.operator)
        return qasm_str
