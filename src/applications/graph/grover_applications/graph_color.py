from qiskit import qasm2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from src.applications.graph.graph_problem import GraphProblem
from src.algorithms.grover import GroverWrapper
from qiskit_algorithms import Grover, AmplificationProblem
from qiskit_aer.primitives import Sampler
from qiskit.visualization import plot_histogram


# variable_qubits = [0, 1, 2, 3, 4, 5, 6, 7]
# check_qubits = [8, 9, 10, 11, 12, 13]
#
# disagree_list = [[[0, 1], [2, 3]],
#                  [[0, 1], [4, 5]],
#                  [[0, 1], [6, 7]],
#                  [[2, 3], [4, 5]],
#                  [[2, 3], [6, 7]],
#                  [[4, 5], [6, 7]]
#                  ]
# output_qubit = 14


def generate_qubit_mapping(nodes, edges):
    # Step 1: Map Nodes to Variable Qubits
    variable_qubits = []
    for node in nodes:
        variable_qubits.extend([2 * node, 2 * node + 1])

    # Step 2: Map Edges to Check Qubits
    check_qubits = list(range(8, 8 + len(edges)))

    # Step 3: Generate the Disagree List
    disagree_list = []
    for i, edge in enumerate(edges):
        node1, node2 = edge
        disagree_list.append([
            [2 * node1, 2 * node1 + 1],
            [2 * node2, 2 * node2 + 1]
        ])

    # Step 4: Assign the Output Qubit
    output_qubit = len(nodes) * 2 + len(edges)  # Assuming output qubit is the next available qubit

    return variable_qubits, check_qubits, disagree_list, output_qubit


def create_equality_checker_circuit(n):
    """
    Create a quantum circuit to check if two n-qubit registers are equal.

    Args:
        n (int): Number of qubits in each register.

    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    # Two n-qubit registers and one check bit, one classical bit for measurement
    qc = QuantumCircuit(3 * n + 1)

    # Registers for qubitsa, qubitsb, and the check bit
    qubits_a = range(n)
    qubits_b = range(n, 2 * n)
    check_bits = range(2 * n, 3 * n)
    output_bit = 3 * n
    # Compare each pair of qubits
    for i in range(n):
        qc.cx(qubits_a[i], qubits_b[i])
        qc.cx(qubits_b[i], check_bits[i])
        #qc.x(check_bits[i])
        # inverse
        qc.cx(qubits_a[i], qubits_b[i])
        qc.barrier()
    # Apply multi-controlled Toffoli gate to set the check bit if all comparisons indicate equality
    qc.mcx([check_bits[i] for i in range(0, n)], output_bit)
    qc.x([check_bits[i] for i in range(0, n)])
    return qc


def graph_color_prep(variable_qubits):
    num_vars = len(variable_qubits)
    prep = QuantumCircuit(num_vars)

    for i in range(num_vars):
        prep.h(i)

    return prep


def disagree_check(qc: QuantumCircuit, qubits_a, qubits_b, check_qubit):
    # 1. Check if the qubits are in the 11 11 state.
    qc.mcx(qubits_a + qubits_b, check_qubit)

    # 2. Check if the qubits are in the 01 01 state.
    qc.x(qubits_a[1])
    qc.x(qubits_b[1])
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a[1])
    qc.x(qubits_b[1])

    # 3. Check if the qubits are in the 10 10 state.
    qc.x(qubits_a[0])
    qc.x(qubits_b[0])
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a[0])
    qc.x(qubits_b[0])

    # 4. Check if the qubits are in the 00 00 state.
    qc.x(qubits_a)
    qc.x(qubits_b)
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a)
    qc.x(qubits_b)


def undo_disagree_check(qc: QuantumCircuit, qubits_a, qubits_b, check_qubit, ancillas):
    # 4. Check if the qubits are in the 00 00 state.
    qc.x(qubits_a)
    qc.x(qubits_b)
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a)
    qc.x(qubits_b)

    # 3. Check if the qubits are in the 10 10 state.
    qc.x(qubits_a[0])
    qc.x(qubits_b[0])
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a[0])
    qc.x(qubits_b[0])

    # 2. Check if the qubits are in the 01 01 state.
    qc.x(qubits_a[1])
    qc.x(qubits_b[1])
    qc.mcx(qubits_a + qubits_b, check_qubit)
    qc.x(qubits_a[1])
    qc.x(qubits_b[1])

    # 1. Check if the qubits are in the 11 11 state.
    qc.mcx(qubits_a + qubits_b, check_qubit)


def graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit):
    # 1. Initializing a quantum circuit with the output bit in the |−⟩ state.
    num_vars = len(variable_qubits)
    num_checks = len(check_qubits)
    num_outputs = 1

    oracle = QuantumCircuit(num_vars + num_checks + num_outputs)
    oracle.x(output_qubit)
    oracle.h(output_qubit)

    # 2. Checking if each pair of qubits in a given list, disagree_list, disagree with each other.
    # Storing the ancilla qubits used for each check
    ancillas = []
    for i in range(len(disagree_list)):
        ancillas += [disagree_check(oracle,
                                    disagree_list[i][0],
                                    disagree_list[i][1],
                                    check_qubits[i])]

    # 3. Flip the output bit if all disagreements are satisfied
    # and also inverse it
    oracle.x(check_qubits)
    oracle.mcx(check_qubits, output_qubit)
    oracle.x(check_qubits)
    # 4. Resetting all the extra qubits for the next iteration.
    # Need to include the specific ancillas used for each check
    for i in range(len(disagree_list)):
        undo_disagree_check(oracle,
                            disagree_list[i][0],
                            disagree_list[i][1],
                            check_qubits[i],
                            ancillas[i])
    #oracle.h(output_qubit)
    #oracle.x(output_qubit)
    return oracle


def check_disagree_list_general(state, disagree_list):
    n = len(state) - 1
    for i in range(len(disagree_list) - 1, -1, -1):
        if (state[n - disagree_list[i][0][0]] == state[n - disagree_list[i][1][0]]
                and state[n - disagree_list[i][0][1]] == state[n - disagree_list[i][1][1]]):
            return False

    return True


class GraphColor(GraphProblem):
    def __init__(self, file_path, verbose=False):
        super().__init__(input_data=file_path)
        self.circuit = None
        variable_qubits, check_qubits, disagree_list, output_qubit = generate_qubit_mapping(self.graph().nodes,
                                                                                            self.graph().edges)
        if verbose:
            print("Variable Qubits:", variable_qubits)
            print("Check Qubits:", check_qubits)
            print("Disagree List:", disagree_list)
            print("Output Qubit:", output_qubit)
        prep = graph_color_prep(variable_qubits)
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)

        # DEFINE THE AmplificationProblem
        def check_disagreement(state): return check_disagree_list_general(state, disagree_list)

        self.iteration = 1
        self.grover_wrapper = GroverWrapper(oracle=oracle,
                                            iterations=self.iteration,
                                            state_preparation=prep,
                                            is_good_state=check_disagreement,
                                            objective_qubits=variable_qubits
                                            )

    def run(self, verbose=False):
        result = self.grover_wrapper.run()
        self.circuit = self.grover_wrapper.grover.construct_circuit(self.grover_wrapper.problem,
                                                                    self.iteration)
        if verbose:
            print(result)
        return result

    def export_to_qasm(self):
        qasm_str = qasm2.dumps(self.circuit)
        return qasm_str
