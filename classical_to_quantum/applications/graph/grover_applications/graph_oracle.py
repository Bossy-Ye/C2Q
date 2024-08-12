from itertools import combinations

import networkx as nx
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
        for qubit_index in negations[-(i + 1)]:
            qc.x(qubit_index)
        qc.append(or_gate.inverse(), clause_qubits + [output_qubit])

        # Apply X gates to revert any negated qubits back to their original state
        for qubit_index in negations[-(i + 1)]:
            qc.x(qubit_index)

    return qc


def cnf_to_quantum_oracle(cnf_formula):
    # Create the initial quantum circuit
    qc = cnf_to_quantum_circuit(cnf_formula)

    # Create an empty quantum circuit to hold the full oracle operations
    qc_tmp = QuantumCircuit(qc.num_qubits)

    # Prepare the final ancilla qubit in the |-> state
    qc_tmp.x(qc.num_qubits - 1)
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.barrier()

    # Append the CNF quantum circuit (the oracle construction)
    qc_tmp.compose(qc, inplace=True)

    # Flip the phase of the ancilla qubit back
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.x(qc.num_qubits - 1)

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


def graph_coloring_to_dimacs(graph: nx.Graph, num_colors: int) -> str:
    """
    Converts a graph coloring problem into a DIMACS-CNF 3-SAT formatted string.

    Parameters:
    - graph (nx.Graph): A NetworkX graph.
    - num_colors (int): Number of colors available for coloring the graph.

    Returns:
    - str: A string formatted in DIMACS-CNF 3-SAT format.
    """
    variables = {}
    clauses = []
    counter = 1

    # Step 1: Create variables x_{v,c} for each vertex v and each color c
    for v in graph.nodes():
        for c in range(1, num_colors + 1):
            variables[(v, c)] = counter
            counter += 1

    # Step 2: Add constraints that each vertex must have at least one color
    for v in graph.nodes():
        clause = [variables[(v, c)] for c in range(1, num_colors + 1)]
        clauses.append(clause)

    # Step 3: Add constraints that no two adjacent vertices share the same color
    for u, v in graph.edges():
        for c in range(1, num_colors + 1):
            clauses.append([-variables[(u, c)], -variables[(v, c)]])

    # Convert the clauses to DIMACS format
    dimacs_str = "c example DIMACS-CNF 3-SAT\n"
    dimacs_str += f"p cnf {counter - 1} {len(clauses)}\n"
    for clause in clauses:
        dimacs_str += " ".join(map(str, clause)) + " 0\n"
    return dimacs_str


import networkx as nx


def maxcut_to_3sat(graph: nx.Graph) -> str:
    """
    Converts a Max-Cut problem into a 3-SAT problem in DIMACS-CNF format.

    Parameters:
    - graph (nx.Graph): A NetworkX graph.

    Returns:
    - str: A string formatted in DIMACS-CNF 3-SAT format.
    """
    variables = {}
    clauses = []
    counter = 1

    # Step 1: Create a boolean variable for each vertex
    for v in graph.nodes():
        variables[v] = counter
        counter += 1

    # Step 2: Add constraints for each edge to ensure the vertices are in different sets
    for u, v in graph.edges():
        u_var = variables[u]
        v_var = variables[v]

        # For 3-SAT, we need to convert each 2-SAT clause into 3-SAT by adding a dummy variable
        dummy_var = counter
        counter += 1

        # (u_var OR v_var) -> (-u_var, -v_var, dummy_var)
        clauses.append([-u_var, -v_var, dummy_var])
        # (NOT u_var OR NOT v_var) -> (u_var, v_var, -dummy_var)
        clauses.append([u_var, v_var, -dummy_var])

    # Convert the clauses to DIMACS format
    dimacs_str = "c Max-Cut to DIMACS-CNF 3-SAT\n"
    dimacs_str += f"p cnf {counter - 1} {len(clauses)}\n"
    for clause in clauses:
        dimacs_str += " ".join(map(str, clause)) + " 0\n"

    return dimacs_str


def independent_set_to_sat(graph: nx.Graph) -> str:
    """
    Converts a Minimum Independent Set problem into a SAT problem in DIMACS-CNF format.

    Parameters:
    - graph (nx.Graph): A NetworkX graph.

    Returns:
    - str: A string formatted in DIMACS-CNF SAT format.
    """
    variables = {}
    clauses = []
    counter = 1

    # Step 1: Create a boolean variable for each vertex
    for v in graph.nodes():
        variables[v] = counter
        counter += 1

    # Step 2: Add constraints for each edge to ensure no two adjacent vertices are both in the independent set
    for u, v in graph.edges():
        u_var = variables[u]
        v_var = variables[v]

        # Add the SAT clause (u OR v)
        clauses.append([-u_var, -v_var])

    # Convert the clauses to DIMACS format
    dimacs_str = "c Minimum Independent Set to DIMACS-CNF SAT\n"
    dimacs_str += f"p cnf {counter - 1} {len(clauses)}\n"
    for clause in clauses:
        dimacs_str += " ".join(map(str, clause)) + " 0\n"

    return dimacs_str


def clique_to_dimacs(graph: nx.Graph, k: int) -> str:
    """
    Converts a Clique problem into a DIMACS-CNF SAT formatted string.

    Parameters:
    - graph (nx.Graph): A NetworkX graph representing the problem.
    - k (int): The size of the clique to find.

    Returns:
    - str: A string in DIMACS-CNF SAT format.
    """
    variables = {}
    clauses = []
    counter = 1

    # Step 1: Create variables x_{iv} where i is the position in the clique (1 to k) and v is a vertex in V
    for i in range(1, k + 1):
        for v in graph.nodes():
            variables[(i, v)] = counter
            counter += 1

    # Step 2: Ensure that each position in the clique is occupied by at least one vertex
    for i in range(1, k + 1):
        clause = [variables[(i, v)] for v in graph.nodes()]
        clauses.append(clause)

    # Step 3: Ensure that the i-th and j-th positions in the clique are occupied by different vertices
    for i in range(1, k):
        for j in range(i + 1, k + 1):
            for v in graph.nodes():
                clauses.append([-variables[(i, v)], -variables[(j, v)]])

    # Step 4: Ensure that any two vertices in the clique are connected
    for i in range(1, k):
        for j in range(i + 1, k + 1):
            for v in graph.nodes():
                for u in graph.nodes():
                    if v != u and not graph.has_edge(v, u):
                        clauses.append([-variables[(i, v)], -variables[(j, u)]])

    # Convert the clauses to DIMACS format
    dimacs_str = "c DIMACS-CNF format for the Clique problem\n"
    dimacs_str += f"p cnf {counter - 1} {len(clauses)}\n"
    for clause in clauses:
        dimacs_str += " ".join(map(str, clause)) + " 0\n"

    return dimacs_str


def vertex_cover_to_dimacs(graph: nx.Graph, k: int) -> str:
    """
    Converts a Vertex Cover problem into a DIMACS-CNF SAT formatted string.

    Parameters:
    - graph (nx.Graph): A NetworkX graph representing the problem.
    - k (int): The size of the vertex cover to find.

    Returns:
    - str: A string in DIMACS-CNF SAT format.
    """
    variables = {}
    clauses = []
    counter = 1

    # Step 1: Create variables x_{iv} where i is the position in the cover (1 to k) and v is a vertex in V
    for i in range(1, k + 1):
        for v in graph.nodes():
            variables[(i, v)] = counter
            counter += 1

    # Step 2: Ensure that each position in the cover is occupied by at least one vertex
    for i in range(1, k + 1):
        clause = [variables[(i, v)] for v in graph.nodes()]
        clauses.append(clause)

    # Step 3: Ensure that no two positions in the cover are occupied by the same vertex
    for v in graph.nodes():
        for i in range(1, k):
            for j in range(i + 1, k + 1):
                clauses.append([-variables[(i, v)], -variables[(j, v)]])

    # Step 4: Ensure that every edge is covered by at least one vertex in the cover
    for u, v in graph.edges():
        clause = []
        for i in range(1, k + 1):
            clause.append(variables[(i, u)])
            clause.append(variables[(i, v)])
        clauses.append(clause)

    # Convert the clauses to DIMACS format
    dimacs_str = "c DIMACS-CNF format for the Vertex Cover problem\n"
    dimacs_str += f"p cnf {counter - 1} {len(clauses)}\n"
    for clause in clauses:
        dimacs_str += " ".join(map(str, clause)) + " 0\n"

    return dimacs_str
