from qiskit_optimization.applications.vertex_cover import VertexCover
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, VQE
from qiskit import QuantumCircuit


class VertexCoverp(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = VertexCover(self.graph())
        qp = self.problem.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        self.qubitOp, self.offset = qubo.to_ising()
