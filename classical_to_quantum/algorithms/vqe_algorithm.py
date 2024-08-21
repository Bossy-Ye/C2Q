# classical_to_quantum/algorithms/vqe_algorithm.py
from qiskit_aer.primitives import Sampler, Estimator

from classical_to_quantum.algorithms.base_algorithm import BaseAlgorithm
from qiskit.circuit.library import TwoLocal, NLocal
from qiskit import QuantumCircuit, qasm2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import SamplingVQE, VQE


class VQEAlgorithm(BaseAlgorithm):
    def __init__(self,
                 observable,
                 n_qubits,
                 reps=3):
        super().__init__()
        self.result = None
        self.ansatz = TwoLocal(n_qubits,
                               rotation_blocks=["ry", "rz"],
                               entanglement_blocks="cz",
                               reps=reps)
        self.observable = observable
        self.n_qubits = n_qubits
        self.reps = reps
        self.vqe = VQE(estimator=Estimator(),
                       ansatz=self.ansatz,
                       optimizer=SPSA(maxiter=100))
        self.is_executed = False

    def run(self, verbose=False):
        """
        Generate and run the quantum code to find the minimum eigenvalue using VQE.
        Try to apply two-local ansatz
        Returns:
            dict: The result containing the eigenvalue and optimal parameters.
        """
        self.result = self.vqe.compute_minimum_eigenvalue(operator=self.observable)
        self.is_executed = True
        if verbose:
            print(self.result)
        return self.result

    def export_to_qasm(self):
        return qasm2.dumps(self.ansatz.bind_parameters(self.result.optimal_point))
