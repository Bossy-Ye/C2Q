# classical_to_quantum/algorithms/base_algorithm.py
class BaseAlgorithm:
    def __init__(self, run_as_classical=False, run_as_quantum=False):
        self.run_as_classical = run_as_classical
        self.run_as_quantum = run_as_quantum
        self.circuit = None

    def generate_quantum_code(self):
        raise NotImplementedError("Subclasses should implement this method")

    def export_to_qasm(self):
        raise NotImplementedError("Subclasses should implement this method")
