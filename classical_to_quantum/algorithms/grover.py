import qiskit
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
                 iteration,
                 state_preparation: QuantumCircuit,
                 is_good_state,
                 objective_qubits
                 ):
        super().__init__()
        self._grover_op = GroverOperator(oracle,
                                         reflection_qubits=objective_qubits)
        self.circuit = state_preparation

        self.circuit.compose(self._grover_op.power(iteration),
                             inplace=True)
        self.problem = AmplificationProblem(oracle,
                                            state_preparation=state_preparation,
                                            is_good_state=is_good_state,
                                            objective_qubits=objective_qubits)
        self.grover = Grover(sampler=Sampler(), iterations=iteration)

    def run(self, verbose=False):
        result = self.grover.amplify(self.problem)
        if verbose:
            print(result)
        return result

    def run_on_quantum(self):
        return None

    def export_to_qasm(self):
        if self.circuit is None:
            raise ValueError("Grover operator has not been generated yet. Call generate_quantum_code() first.")
        qasm_str = qiskit.qasm2.dumps(self.circuit)
        return qasm_str
