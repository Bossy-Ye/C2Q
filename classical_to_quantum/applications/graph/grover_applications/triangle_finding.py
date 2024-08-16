import math

import numpy as np
import qiskit
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover

from classical_to_quantum.applications.graph.graph_problem import GraphProblem
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit import qasm3
from qiskit import qasm2
from qiskit.primitives import Sampler
from classical_to_quantum.algorithms.grover import GroverWrapper


# We used the W state implementation from W state in reference 6
def control_rotation(qcir, cQbit, tQbit, theta):
    """ Create an intermediate controlled rotation using only unitary gate and controlled-NOT

    Args:
    qcir: QuantumCircuit instance to apply the controlled rotation to.
    cQbit: control qubit.
    tQbit: target qubit.
    theta: rotation angle.

    Returns:
    A modified version of the QuantumCircuit instance with control rotation applied.

    """
    theta_dash = math.asin(math.cos(math.radians(theta / 2)))
    qcir.u(theta_dash, 0, 0, tQbit)
    qcir.cx(cQbit, tQbit)
    qcir.u(-theta_dash, 0, 0, tQbit)
    return qcir


def wn(qcir, qbits):
    """ Create the W-state using the control-rotation function.

    Args:
    qcir: QuantumCircuit instance used to construct the W-state.
    qbits: the qubits used to construct the W-state.

    Returns:
    A modified version of the QuantumCircuit instance with the W-state construction gates.

    """
    for i in range(len(qbits)):
        if i == 0:
            qcir.x(qbits[0])
            qcir.barrier()
        else:
            p = 1 / (len(qbits) - (i - 1))
            theta = math.degrees(math.acos(math.sqrt(p)))
            theta = 2 * theta
            qcir = control_rotation(qcir, qbits[i - 1], qbits[i], theta)
            qcir.cx(qbits[i], qbits[i - 1])
            qcir.barrier()
    return qcir, qbits


def edge_counter(qc, qubits, anc, flag_qubit, k):
    bin_k = bin(k)[2:][::-1]
    l = []
    for i in range(len(bin_k)):
        if int(bin_k[i]) == 1:
            l.append(qubits[i])
    qc.mcx(l, flag_qubit, [anc])


def oracle(n_nodes, edges, qc, nodes_qubits, edge_anc, ancilla, neg_base):
    k = 3  # k is the number of edges, in case of a triangle, it's 3
    # 1- edge counter
    # forward circuit
    qc.barrier()
    qc.ccx(nodes_qubits[edges[0][0]], nodes_qubits[edges[0][1]], edge_anc[0])
    for i in range(1, len(edges)):
        qc.mcx([nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0]], edge_anc[1], [ancilla[0]])
        qc.ccx(nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0])
    # ----------------------------------------------------------------------------------------------------------
    # Edges check Qubit
    edg_k = int((k / 2) * (k - 1))
    edge_counter(qc, edge_anc, ancilla[0], neg_base[0], edg_k)
    # ----------------------------------------------------------------------------------------------------------

    # 4- Reverse edge count
    for i in range(len(edges) - 1, 0, -1):
        qc.ccx(nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0])
        qc.mcx([nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0]], edge_anc[1], [ancilla[0]])
    qc.ccx(nodes_qubits[edges[0][0]], nodes_qubits[edges[0][1]], edge_anc[0])
    qc.barrier()


def triangle_oracle(edges, nodes_qubits, edge_anc, ancilla, neg_base):
    k = 3  # k is the number of edges, in case of a triangle, it's 3
    # 1- edge counter
    # forward circuit
    qc = QuantumCircuit(nodes_qubits, edge_anc, ancilla, neg_base)
    qc.x(neg_base[0])
    qc.h(neg_base[0])
    qc.barrier()
    qc.ccx(nodes_qubits[edges[0][0]], nodes_qubits[edges[0][1]], edge_anc[0])
    for i in range(1, len(edges)):
        qc.mcx([nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0]], edge_anc[1], [ancilla[0]])
        qc.ccx(nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0])
    # ----------------------------------------------------------------------------------------------------------
    # Edges check Qubit
    edg_k = int((k / 2) * (k - 1))
    edge_counter(qc, edge_anc, ancilla[0], neg_base[0], edg_k)
    # ----------------------------------------------------------------------------------------------------------

    # 4- Reverse edge count
    for i in range(len(edges) - 1, 0, -1):
        qc.ccx(nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0])
        qc.mcx([nodes_qubits[edges[i][0]], nodes_qubits[edges[i][1]], edge_anc[0]], edge_anc[1], [ancilla[0]])
    qc.ccx(nodes_qubits[edges[0][0]], nodes_qubits[edges[0][1]], edge_anc[0])
    qc.barrier()

    qc.h(neg_base[0])
    qc.x(neg_base[0])
    return qc


def cnz(qc, num_control, node, anc):
    """Construct a multi-controlled Z gate

    Args:
    num_control :  number of control qubits of cnz gate
    node :             node qubits
    anc :               ancillaly qubits
    """
    #
    if num_control > 2:
        qc.ccx(node[0], node[1], anc[0])
        for i in range(num_control - 2):
            qc.ccx(node[i + 2], anc[i], anc[i + 1])
        qc.cz(anc[num_control - 2], node[num_control])
        for i in range(num_control - 2)[::-1]:
            qc.ccx(node[i + 2], anc[i], anc[i + 1])
        qc.ccx(node[0], node[1], anc[0])
    if num_control == 2:
        qc.h(node[2])
        qc.ccx(node[0], node[1], node[2])
        qc.h(node[2])
    if num_control == 1:
        qc.cz(node[0], node[1])


def grover_diff(qc, nodes_qubits, edge_anc, ancilla, stat_prep, inv_stat_prep):
    qc.append(inv_stat_prep, qargs=nodes_qubits)
    qc.x(nodes_qubits)
    #====================================================
    #3 control qubits Z gate
    cnz(qc, len(nodes_qubits) - 1, nodes_qubits[::-1], ancilla)
    #====================================================
    qc.x(nodes_qubits)
    qc.append(stat_prep, qargs=nodes_qubits)


# Grover algo function
def grover(n_nodes, stat_prep, inv_stat_prep, edges):
    N = math.comb(n_nodes, 3)
    nodes_qubits = QuantumRegister(n_nodes, name='nodes')
    edge_anc = QuantumRegister(2, name='edge_anc')
    ancilla = QuantumRegister(n_nodes - 2, name='cccx_diff_anc')
    neg_base = QuantumRegister(1, name='check_qubits')
    class_bits = ClassicalRegister(n_nodes, name='class_reg')
    tri_flag = ClassicalRegister(3, name='tri_flag')
    qc = QuantumCircuit(nodes_qubits, edge_anc, ancilla, neg_base, class_bits, tri_flag)
    # Initialize qunatum flag qubits in |-> state
    qc.x(neg_base[0])
    qc.h(neg_base[0])
    # Initializing i/p qubits in superposition
    qc.append(stat_prep, qargs=nodes_qubits)
    qc.barrier()
    # Calculate iteration count
    iterations = math.floor(math.pi / 4 * math.sqrt(N))
    # Calculate iteration count
    for i in np.arange(iterations):
        qc.barrier()
        oracle(n_nodes, edges, qc, nodes_qubits, edge_anc, ancilla, neg_base)
        qc.barrier()
        grover_diff(qc, nodes_qubits, edge_anc, ancilla, stat_prep, inv_stat_prep)
    qc.measure(nodes_qubits, class_bits)
    return qc


new_node_id = 0


class TriangleFinding(GraphProblem):

    def __init__(self, file_path):
        super().__init__(file_path)
        self.circuit = None
        self.grover = None
        self.operator = None
        node_map = {}

        def get_new_node_id(old_node):
            global new_node_id
            if old_node not in node_map:
                node_map[old_node] = new_node_id
                new_node_id += 1
            return node_map[old_node]

        # Extract the edges with the new node IDs, checking weights
        self.edges = []
        global new_node_id
        new_node_id = 0
        for (u, v, w) in self.elist:
            if w != 1.0:
                raise ValueError(f"Edge ({u}, {v}) has a weight {w} which is not 1.0")
            new_u = get_new_node_id(u)
            new_v = get_new_node_id(v)
            self.edges.append((new_u, new_v))

        # define oracle and prep circuits
        n_nodes = self.num_nodes
        N = 2**n_nodes
        self.iterations = math.floor(math.pi / 4 * math.sqrt(N))
        nodes_qubits = QuantumRegister(n_nodes, name='nodes')
        edge_anc = QuantumRegister(2, name='edge_anc')
        ancilla = QuantumRegister(n_nodes - 2, name='cccx_diff_anc')
        neg_base = QuantumRegister(1, name='check_qubits')

        self.oracle = triangle_oracle(self.edges, nodes_qubits, edge_anc, ancilla, neg_base)
        sub_qbits = QuantumRegister(n_nodes)
        prep = QuantumCircuit(sub_qbits, name="state_prep")
        prep, sub_qbits = wn(prep, sub_qbits)
        prep.x(sub_qbits)
        self.prep = prep

        def check_valid_triangle(state):
            nodes = []
            for i, value in enumerate(state):
                if value == '1':
                    nodes.append(self.num_nodes - 1 - i)
            if len(nodes) != 3:
                return False
            a, b, c = nodes
            edge_set = set()
            for edge in self.edges:
                edge_set.add(tuple(sorted(edge)))

            # Check if all three edges of the triangle are present
            return (
                    tuple(sorted([a, b])) in edge_set and
                    tuple(sorted([b, c])) in edge_set and
                    tuple(sorted([c, a])) in edge_set
            )

        self.grover_wrapper = GroverWrapper(oracle=self.oracle,
                                            iterations=self.iterations,
                                            state_preparation=self.prep,
                                            is_good_state=check_valid_triangle,
                                            objective_qubits=list(range(self.num_nodes))
                                            )

    def search(self):
        result = self.grover_wrapper.run()
        self.circuit = self.grover_wrapper.grover.construct_circuit(self.grover_wrapper.problem,
                                                                    self.iteration)
        return result

    def export_to_qasm(self):
        qasm_str = qasm2.dumps(self.circuit)
        return qasm_str
