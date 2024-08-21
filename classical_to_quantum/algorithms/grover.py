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
from qiskit import qasm2
from qiskit.primitives import Sampler


class GroverWrapper(BaseAlgorithm):
    def __init__(self,
                 oracle: QuantumCircuit,
                 iterations,
                 is_good_state = None,
                 state_preparation: QuantumCircuit = None,
                 objective_qubits=None
                 ):
        super().__init__()
        self._grover_op = GroverOperator(oracle,
                                         reflection_qubits=objective_qubits)
        if objective_qubits is None:
            objective_qubits = list(range(oracle.num_qubits))
        if state_preparation is None:
            state_preparation = QuantumCircuit(oracle.num_qubits)
            state_preparation.h(objective_qubits)
        if is_good_state is None:
            def func(state):
                return True
            is_good_state = func
        self.circuit = QuantumCircuit(oracle.num_qubits, len(objective_qubits))
        self.circuit.compose(state_preparation, inplace=True)
        self.circuit.compose(self._grover_op.power(iterations),
                             inplace=True)
        self.circuit.measure(objective_qubits,
                             objective_qubits)
        self.problem = AmplificationProblem(oracle,
                                            state_preparation=state_preparation,
                                            is_good_state=is_good_state,
                                            objective_qubits=objective_qubits)
        self.grover = Grover(sampler=Sampler(), growth_rate=1.25)

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
        qasm = self.circuit.qasm()
        #TODO error with c3sqrtx
        #qasm = qasm.replace("c3sqrtx", "c3sx")
        return qasm
