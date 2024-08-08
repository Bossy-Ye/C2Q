from qiskit_optimization.applications.exact_cover import ExactCover
from classical_to_quantum.applications.graph.graph_problem import GraphProblem
from qiskit_optimization.converters import QuadraticProgramToQubo


class ExactCoverp(GraphProblem):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = ExactCover(self.graph())
        self.qp = self.problem.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(self.qp)
        self.qubitOp, self.offset = qubo.to_ising()


