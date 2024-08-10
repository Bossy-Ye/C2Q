# classical_to_quantum/algorithms/vqe_algorithm.py

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


class VQEAlgorithm(BaseAlgorithm):
    def __init__(self, observable: object, n_qubits,
                 run_as_classical=True, run_as_quantum=False,
                 reps=3):
        super().__init__(run_as_classical, run_as_quantum)
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
        if self.run_as_classical:
            estimator = StatevectorEstimator()

            x0 = np.ones(len(ansatz.parameters))
            result = minimize(cost_func_vqe, x0,
                              args=(ansatz, self.observable, estimator),
                              method="COBYLA")
            self.result = result
            if verbose:
                print(self.result)
        elif self.run_as_quantum:
            service = QiskitRuntimeService(channel='ibm_quantum')
            backend = service.least_busy(operational=True, simulator=False)

            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            isa_ansatz = pm.run(ansatz)
            isa_observable = self.observable.apply_layout(layout=isa_ansatz.layout)

            x0 = np.ones(len(ansatz.parameters))

            with Session(backend=backend) as session:
                session_options = Options()
                session_options.execution.shots = 4096
                session_options.resilience_level = 1

                estimator = Estimator(mode=session)
                sampler = Sampler(mode=session)
                estimator.options.default_shots = 10_000
                result = minimize(cost_func_vqe, x0,
                                  args=(isa_ansatz, isa_observable, estimator),
                                  method="COBYLA")

                self.result = result

            session.close()

    def export_to_qasm(self):
        return qasm3.dumps(self.ansatz.assign_parameters(self.result.x),
                           experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1)
