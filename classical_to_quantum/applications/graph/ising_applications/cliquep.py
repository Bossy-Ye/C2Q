from classical_to_quantum.applications.graph.gset import *
from qiskit import qasm3
from qiskit_optimization.applications.clique import Clique
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising


class CliqueP(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = Clique(self.graph())
        self.qp = self.problem.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(self.qp)
        self.qubitOp, self.offset = qubo.to_ising()
