from itertools import combinations

import networkx as nx
import qiskit.qasm2
from matplotlib import pyplot as plt
from pysat.formula import CNF
from qiskit.circuit.library import PhaseOracle

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from pysat.formula import CNF
import matplotlib.pyplot as plt
# Example CNF Formula using PySAT
from qiskit.circuit.library import OR
from qiskit.quantum_info import Statevector
from utils import get_evolved_state


def cnf_to_quantum_circuit(cnf_formula):
    num_vars = cnf_formula.nv  # Number of variables
    num_clauses = len(cnf_formula.clauses)

    # Create a quantum circuit with num_vars input qubits, num_clauses output qubits, and 1 final ancilla qubit
    qc = QuantumCircuit(num_vars + num_clauses + 1)  # +1 for the final ancilla

    # List to keep track of output qubits for each clause
    clause_outputs = []
    or_gates = []  # To keep track of the OR gates for uncomputing
    negations = []  # To keep track of which qubits were negated

    # Step through each clause and create an OR gate
    for i, clause in enumerate(cnf_formula.clauses):
        clause_qubits = []
        clause_negations = []
        for lit in clause:
            qubit_index = abs(lit) - 1  # Convert variable index to qubit index
            if lit < 0:  # If the literal is negative (~x)
                qc.x(qubit_index)  # Apply X to flip qubit before using in OR
                clause_negations.append(qubit_index)
            clause_qubits.append(qubit_index)

        # OR gate for the clause, outputting to an ancillary qubit
        or_gate = OR(len(clause_qubits))
        output_qubit = num_vars + i  # Assign output qubit from the ancillary pool
        clause_outputs.append(output_qubit)

        # Apply the OR gate: inputs are the qubits from the clause, output is the clause output qubit
        qc.append(or_gate, clause_qubits + [output_qubit])

        if clause_negations:
            qc.x(clause_negations)

        # Add a barrier after applying the OR gate and any necessary re-negations
        qc.barrier()

        # Store the OR gate and negations for uncomputing
        or_gates.append((or_gate, clause_qubits, output_qubit))
        negations.append(clause_negations)

    # Combine all clause outputs into the final oracle condition
    qc.mcx(clause_outputs, num_vars + num_clauses)

    # Add a barrier after applying the multi-controlled X gate
    qc.barrier()

    # Uncompute the OR gates by applying them in reverse order
    for i, (or_gate, clause_qubits, output_qubit) in enumerate(reversed(or_gates)):
        qc.barrier()
        for qubit_index in negations[-(i + 1)]:
            qc.x(qubit_index)
        qc.append(or_gate.inverse(), clause_qubits + [output_qubit])

        # Apply X gates to revert any negated qubits back to their original state
        for qubit_index in negations[-(i + 1)]:
            qc.x(qubit_index)
    qc.barrier()
    return qc


def cnf_to_quantum_oracle(cnf_formula):
    # Create the initial quantum circuit
    qc = cnf_to_quantum_circuit(cnf_formula)

    # Create an empty quantum circuit to hold the full oracle operations
    qc_tmp = QuantumCircuit(qc.num_qubits)

    qc_tmp.barrier()
    # Prepare the final ancilla qubit in the |-> state
    qc_tmp.x(qc.num_qubits - 1)
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.barrier()

    # Append the CNF quantum circuit (the oracle construction)
    qc_tmp.compose(qc, inplace=True)

    # Flip the phase of the ancilla qubit back
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.x(qc.num_qubits - 1)

    qc_tmp.barrier()
    return qc_tmp


def graph_coloring_to_sat(graph: nx.Graph, num_colors: int) -> CNF:
    """
    Converts a graph coloring problem into a 3-SAT problem.

    Parameters:
    - graph (nx.Graph): A networkx graph.
    - num_colors (int): Number of colors available for coloring the graph.

    Returns:
    - CNF: A CNF object representing the 3-SAT problem.
    """
    cnf = CNF()
    variables = {}
    counter = 1

    # Step 1: Create variables x_{v,c} for each vertex v and each color c
    for v in graph.nodes():
        for c in range(1, num_colors + 1):
            variables[(v, c)] = counter
            counter += 1

    # Step 2: Add constraints that each vertex must have at least one color
    for v in graph.nodes():
        clause = [variables[(v, c)] for c in range(1, num_colors + 1)]
        cnf.append(clause)

    # Step 3: Add constraints that no two adjacent vertices share the same color
    for u, v in graph.edges():
        for c in range(1, num_colors + 1):
            cnf.append([-variables[(u, c)], -variables[(v, c)]])

    return cnf


def independent_set_to_sat(graph: nx.Graph) -> CNF:
    """
    Converts an Independent Set problem into a SAT problem in CNF format.

    Parameters:
    - graph (nx.Graph): A NetworkX graph.

    Returns:
    - CNF: A CNF object representing the SAT problem.
    """
    cnf = CNF()
    n = len(graph.nodes)

    # Variables: x_i where i is the vertex index
    var = lambda i: i + 1  # Variables indexed from 1

    # Clause 1: For every edge (u, v), add a clause that at least one of them must not be in the independent set
    for u, v in graph.edges():
        cnf.append([-var(u), -var(v)])

    return cnf


def clique_to_sat(graph: nx.Graph, k: int) -> CNF:
    """
    Converts the k-Clique problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.
        k (int): The size of the clique to find.

    Returns:
        CNF: The SAT formula in CNF representing the k-Clique problem.
    """
    cnf = CNF()
    n = len(graph.nodes)

    # Variables: x_ij where i is the vertex index and j is the position in the clique
    var = lambda i, j: i * k + j + 1  # Create unique variables

    # Clause 1: Each position in the clique must be occupied by at least one vertex
    for j in range(k):
        cnf.append([var(i, j) for i in range(n)])

    # Clause 2: No vertex can occupy more than one position in the clique
    for i in range(n):
        for j in range(k):
            for jp in range(j + 1, k):
                cnf.append([-var(i, j), -var(i, jp)])

    # Clause 3: No two vertices in the clique can be non-adjacent
    for i in range(n):
        for ip in range(i + 1, n):
            if not graph.has_edge(i, ip):
                for j in range(k):
                    cnf.append([-var(i, j), -var(ip, j)])

    return cnf

