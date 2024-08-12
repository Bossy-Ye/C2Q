import math
import unittest

import matplotlib.pyplot as plt
from qiskit.circuit.library import GroverOperator
from qiskit.primitives import Sampler
from qiskit_algorithms import AmplificationProblem, Grover

from algorithms.grover import GroverWrapper
from classical_to_quantum.applications.graph.grover_applications.graph_oracle import *
from pysat.formula import CNF
from pysat.solvers import Solver
from qiskit.visualization import plot_histogram


def adjust_expected_states(expected_states, num_vars, total_qubits):
    """
    Adjusts the expected states to match the total qubits by appending zeros to the end.
    The extra qubits represent the clause output qubits and the final ancilla qubit.
    """
    adjusted_states = []
    for state in expected_states:
        adjusted_state = '0' * (total_qubits - num_vars) + state
        adjusted_states.append(adjusted_state)
    return adjusted_states


def solve_all_cnf_solutions(cnf_formula):
    """
    Finds all solutions to the CNF formula using a SAT solver.

    Args:
        cnf_formula (CNF): The CNF formula to solve.

    Returns:
        list: A list of all satisfying assignments, where each assignment is a list of literals.
    """
    solutions = []
    with Solver(bootstrap_with=cnf_formula) as solver:
        for model in solver.enum_models():
            solutions.append(model)
    return solutions


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_cnf_circuit(self):
        test_cases = [
            (CNF(from_clauses=[[1, -2], [-1, 3], [2]]), ['000', '100', '101', '111', '010']),  # Test Case 1
            (CNF(from_clauses=[[1, 2], [2, 3]]), ['010', '110', '111', '101', '011']),  # Test Case 2
            (CNF(from_clauses=[[1, 2, -3], [-1, 2, -4], [1, -2, 3, 5]]),
             ['01011', '10101', '00000', '00110', '01110']),  # Test Case 4
        ]
        for cnf_formula, expected_states in test_cases:
            print(f'-----------{cnf_formula}-----------------')
            solutions = solve_all_cnf_solutions(cnf_formula)
            print(f"Solution for CNF3: {solutions}")
            #circuit = cnf_to_quantum_circuit(cnf_formula)
            circuit = cnf_to_quantum_oracle(cnf_formula)
            total_qubits = circuit.num_qubits
            adjusted_states = adjust_expected_states(expected_states, cnf_formula.nv, total_qubits)
            for state_label in adjusted_states:
                state = Statevector.from_label(state_label)
                get_evolved_state(circuit, state, verbose=True)

    def test_quantum_oracle_cnf(self):
        test_cases = [
            (CNF(from_clauses=[[1, -2], [-1, 3], [2]]), ['000', '100', '101', '111', '010']),  # Test Case 1
            (CNF(from_clauses=[[1, 2], [2, 3]]), ['010', '110', '111', '101', '011']),  # Test Case 2
            (CNF(from_clauses=[[1, 2, -3], [-1, 2, -4], [1, -2, 3, 5]]),
             ['01011', '10101', '00000', '00110', '01110']),  # Test Case 4
        ]
        for cnf_formula, expected_states in test_cases:
            print(f'-----------{cnf_formula}-----------------')
            solutions = solve_all_cnf_solutions(cnf_formula)
            print(f"Solution for CNF3: {solutions}")
            #circuit = cnf_to_quantum_circuit(cnf_formula)
            oracle = cnf_to_quantum_oracle(cnf_formula)
            oracle.barrier()
            grover_op = GroverOperator(oracle, reflection_qubits=[0, 1, 2])
            optimal_num_iterations = math.floor(
                math.pi / (4 * math.asin(math.sqrt(5 / 2 ** grover_op.num_qubits)))
            )
            qc = QuantumCircuit(grover_op.num_qubits)
            # Create even superposition of all basis states
            qc.h(range(cnf_formula.nv))
            print(grover_op.decompose())
            # Apply Grover operator the optimal number of times
            qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
            # Measure all qubits
            qc.measure_all()
            print(qc)
            sampler = Sampler()
            result = sampler.run(qc, shots=1000)
            # Access the first quasi-probability distribution
            # Access the first quasi-probability distribution
            quasi_dists = result.result().quasi_dists[0]
            # Filter out only the first 3 bits for plotting
            filtered_counts = {}
            for state, probability in quasi_dists.items():
                # Take only the first 3 bits of the state
                binary_state = format(state, f'0{grover_op.num_qubits}b')
                filtered_state = binary_state[:3]
                if filtered_state in filtered_counts:
                    filtered_counts[filtered_state] += probability
                else:
                    filtered_counts[filtered_state] = probability
            # Plot the histogram for the first 3 bits
            plot_histogram(filtered_counts)
            plt.show()

    def test_clique_grover(self):
        # Example usage:
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
        k = 3

        dimacs_string = clique_to_dimacs(G, 3)
        print(dimacs_string)

        import os
        import tempfile
        from qiskit.exceptions import MissingOptionalLibraryError
        from qiskit.circuit.library.phase_oracle import PhaseOracle
        from classical_to_quantum.algorithms.grover import GroverWrapper

        fp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        fp.write(dimacs_string)
        file_name = fp.name
        fp.close()
        oracle = None
        try:
            oracle = PhaseOracle.from_dimacs_file(file_name)
        except ImportError as ex:
            print(ex)
        finally:
            os.remove(file_name)

        grover = GroverWrapper(oracle,
                               is_good_state=oracle.evaluate_bitstring,
                               iteration=1)
        grover.run(verbose=True)

    def test_grover_qiskit(self):
        # Define the CNF formula
        cnf_formula = CNF(from_clauses=[[1, -2], [-1, 3], [2]])

        # Convert the CNF formula to a quantum oracle
        oracle_circuit = cnf_to_quantum_oracle(cnf_formula)

        print(oracle_circuit)


if __name__ == '__main__':
    unittest.main()
