from qiskit_optimization.applications.vehicle_routing import VehicleRouting
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, VQE
from Ising import Ising


class VehicleRoutep(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = VehicleRouting(self.graph(), 2, 0)
        qp = self.problem.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        self.qubitOp, self.offset = qubo.to_ising()
