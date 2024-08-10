from classical_to_quantum.algorithms.base_algorithm import BaseAlgorithm
from qiskit.circuit.library import TwoLocal, NLocal
from qiskit import QuantumCircuit
import numpy as np
from qiskit.primitives import StatevectorEstimator as Estimator, StatevectorEstimator
from qiskit.primitives import StatevectorSampler as Sampler
# SciPy minimizer routine
from scipy.optimize import minimize
import time
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import Session, Options
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qasm3
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms import VQD


class VQDAlgorithm(BaseAlgorithm):
    def __init__(self, observable: object, n_qubits, n_eigenvectors: int,
                 run_as_classical=True, run_as_quantum=False,
                 reps=3):
        super().__init__(run_as_classical, run_as_quantum)
        self.ansatz = None
        self.observable = observable
        self.n_eigenvectors = n_eigenvectors
        self.n_qubits = n_qubits
        self.reps = reps
        self.resulted_circuit = None

    def run(self):
        """
        Generate and run the quantum code to find the n_eigenvectors using VQD.
        Try to apply two-local ansatz
        Returns:
            dict: The result containing the eigenvalue and optimal parameters.
            Or None if ====
        """
        ansatz = TwoLocal(self.n_qubits, rotation_blocks=["ry", "rz"],
                          entanglement_blocks="cz", reps=self.reps)
        optimizer = SLSQP()
        estimator = Estimator()
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler)

        betas = [33] * self.n_eigenvectors

        counts = []
        values = []
        steps = []

        def callback(eval_count, params, value, meta, step):
            counts.append(eval_count)
            values.append(value)
            steps.append(step)

        if self.run_as_classical:
            vqd = VQD(estimator, fidelity, ansatz, optimizer,
                      k=self.n_eigenvectors, betas=betas, callback=callback)
            result = vqd.compute_eigenvalues(operator=self.observable)
            self.resulted_circuit = result.optimal_circuits

    def export_to_qasm(self):
        qasm3_strings = [qasm3.dumps(circuit) for circuit in self.resulted_circuit]
        merged_qasm3 = '\n'.join(qasm3_strings)
        return merged_qasm3
