from classical_to_quantum.algorithms.base_algorithm import BaseAlgorithm



class QAOAlgorithm(BaseAlgorithm):
    def __init__(self, matrix : object,
                 run_as_classical=True, run_as_quantum=False,
                 ):
        super().__init__(run_as_classical, run_as_quantum)

    def generate_quantum_code(self):
        """
        Generate and run the quantum code to find the n_eigenvectors using VQD.
        Try to apply two-local ansatz
        Returns:
            dict: The result containing the eigenvalue and optimal parameters.
            Or None if ====
        """
    def export_to_qasm3(self):
        return

