import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from pysat.formula import CNF
import matplotlib.pyplot as plt
# Example CNF Formula using PySAT
from qiskit.circuit.library import OR
from qiskit.quantum_info import Statevector
from utils import get_evolved_state
from graph_oracle import cnf_to_quantum_oracle

def cnf_to_oracle_with_or(cnf_formula):
    num_vars = cnf_formula.nv  # Number of variables
    num_clauses = len(cnf_formula.clauses)

    # Create a quantum circuit with num_vars input qubits, num_clauses output qubits, and 1 final ancilla qubit
    qc = QuantumCircuit(num_vars + num_clauses + 1)  # +1 for the final ancilla

    # List to keep track of output qubits for each clause
    clause_outputs = []
    or_gates = []  # To keep track of the OR gates for uncomputing
    negations = []  # To keep track of which qubits were negated for uncomputing later...

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


test_cases = [
    (CNF(from_clauses=[[1, -2], [-1, 3]]), ['011', '100', '101']),  # Test Case 1
    (CNF(from_clauses=[[1, 2], [2, 3]]), ['000', '001', '010', '011', '100', '101', '110']),  # Test Case 2
    (CNF(from_clauses=[[1, 2, -3]]), ['000', '001', '010', '011', '100', '101', '110']),  # Test Case 4
]

cnf = CNF(from_clauses=[
    [1, 2, 3],  # Clause 1: (x1 ∨ ¬x2 ∨ x3)
    [-1, 2, 4],  # Clause 2: (¬x1 ∨ x2 ∨ x4)
    [2, 3, -5]
])

# Create the circuit and check the last bit
circuit = QuantumCircuit(9)
circuit.h([0, 1, 2, 3, 4])
state = Statevector(circuit)
oracle = cnf_to_quantum_oracle(cnf)
get_evolved_state(oracle, state, verbose=True)
