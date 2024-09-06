# src/algorithms/base_algorithm.py
class BaseAlgorithm:
    def __init__(self):
        self.circuit = None

    def run(self):
        """
        run on simulator
        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method")

    def run_on_quantum(self):
        """
        run on quantum computer
        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method")

    def export_to_qasm(self):
        raise NotImplementedError("Subclasses should implement this method")
