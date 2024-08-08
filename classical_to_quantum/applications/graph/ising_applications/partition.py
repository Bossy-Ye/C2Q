from classical_to_quantum.applications.graph.gset import *
from qiskit import qasm3
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising
from qiskit_optimization.applications.graph_partition import GraphPartition
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import (COBYLA, SPSA,
                                          L_BFGS_B, SLSQP,
                                          ADAM)
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, VQE
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator, Aer


class Partition(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = GraphPartition(self.graph())
        self.qp = self.problem.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(self.qp)
        self.qubitOp, self.offset = qubo.to_ising()

