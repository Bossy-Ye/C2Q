# classical_to_quantum/algorithms/vqe_algorithm.py

from classical_to_quantum.algorithms.base_algorithm import BaseAlgorithm
from qiskit.circuit.library import TwoLocal, NLocal
from qiskit import QuantumCircuit, qasm2
import numpy as np
# SciPy minimizer routine
from scipy.optimize import minimize
import time
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qasm3
from qiskit.primitives import StatevectorEstimator


class VQEAlgorithm(BaseAlgorithm):
    def __init__(self, observable: object, n_qubits,
                 reps=3):
        super().__init__()
        self.result = None
        self.ansatz = None
        self.observable = observable
        self.n_qubits = n_qubits
        self.reps = reps

    def run(self, verbose=False):
        """
        Generate and run the quantum code to find the minimum eigenvalue using VQE.
        Try to apply two-local ansatz
        Returns:
            dict: The result containing the eigenvalue and optimal parameters.
        """

        def cost_func_vqe(params, ansatz, hamiltonian, estimator):
            """Return estimate of energy from estimator

            Parameters:
                params (ndarray): Array of ansatz parameters
                ansatz (QuantumCircuit): Parameterized ansatz circuit
                hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
                estimator (Estimator): Estimator primitive instance

            Returns:
                float: Energy estimate
            """
            pub = (ansatz, hamiltonian, params)
            cost = estimator.run([pub]).result()[0].data.evs

            return cost

        reference_circuit = QuantumCircuit(self.n_qubits)
        reference_circuit.x(0)
        variational_form = TwoLocal(
            self.n_qubits,
            rotation_blocks=["rz", "ry"],
            entanglement_blocks="cx",
            entanglement="linear",
            reps=self.reps,
        )

        ansatz = reference_circuit.compose(variational_form)

        ansatz.decompose().draw('mpl')
        self.ansatz = ansatz
        self.circuit = ansatz
        # classical estimator below
        estimator = StatevectorEstimator()

        x0 = np.ones(len(ansatz.parameters))
        result = minimize(cost_func_vqe, x0,
                            args=(ansatz, self.observable, estimator),
                            method="COBYLA")
        self.result = result
        if verbose:
            print(self.result)

    def export_to_qasm(self):
        return qasm2.dumps(self.ansatz.assign_parameters(self.result.x))
