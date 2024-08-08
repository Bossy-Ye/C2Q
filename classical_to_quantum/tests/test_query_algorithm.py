from qiskit.circuit import QuantumCircuit
import qiskit.qasm3 as qasm3
from qiskit.visualization import circuit_drawer
qc = qasm3.loads("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nh q[0];")


circuit_drawer(qc, output="mpl", filename="test_query_algorithm.png")
