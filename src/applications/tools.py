from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector


class OptimizerLog:
    def __init__(self):
        self._values = []
        self._counts = []
        self._params = []

    def update(self, _nfevs, _theta, ftheta, *_):
        """
        nfevs: number of function evaluations
        theta: theta vector
        ftheta: function evaluations
        *_: metadata
        """
        self._counts.append(_nfevs)
        self._params.append(_theta)
        self._values.append(ftheta)

    def get_values(self):
        return self._values

    def get_params(self):
        return self._params

    def get_counts(self):
        return self._counts



