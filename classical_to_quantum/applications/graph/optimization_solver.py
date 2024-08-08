import matplotlib.pyplot as plt

from docplex.mp.model import Model

from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import (COBYLA, SPSA,
                                          L_BFGS_B, SLSQP,
                                          ADAM, NFT, AQGD, CG, GSLS)
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, Estimator
from qiskit_optimization.algorithms import (CobylaOptimizer,
                                            MinimumEigenOptimizer, GroverOptimizer)
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_aer import Aer
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient, ParamShiftSamplerGradient


class OptSolver:
    """
    provide appropriate optimizer for various optimization problems
    1. QAOA
    2. VQE
    3. GROVER
    """
    optimizers = [
        COBYLA(),
        SPSA(maxiter=300, learning_rate=None, perturbation=None),
        #L_BFGS_B(),
        #SLSQP(maxiter=500),
        #SPSA()
    ]
    gradient = [
        FiniteDiffEstimatorGradient(estimator=Estimator(), epsilon=0.01),
    ]

    def __init__(self):
        self._qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=1000),
                          reps=3)

    def getQAOA(self):
        return self._qaoa

    @staticmethod
    def getVQE(num_qubits):
        two_local = TwoLocal(num_qubits=num_qubits,
                             rotation_blocks=['ry'],
                             entanglement_blocks='cz',
                             entanglement='linear',
                             reps=2,
                             skip_unentangled_qubits=False,
                             insert_barriers=True)

        return SamplingVQE(sampler=Sampler(), optimizer=SPSA(maxiter=200),
                           ansatz=two_local)

    @staticmethod
    def getGroverOptimizer(self, num_qubits, num_iterations=10):
        return GroverOptimizer(num_value_qubits=num_qubits,
                               num_iterations=num_iterations, sampler=Sampler())
