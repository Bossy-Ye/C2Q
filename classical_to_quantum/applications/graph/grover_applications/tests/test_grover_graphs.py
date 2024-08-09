import unittest
from classical_to_quantum.applications.graph.grover_applications.graph_color import *
from qiskit.quantum_info import Statevector
from classical_to_quantum.applications.graph.grover_applications.triangle_finding import *

def print_amplitudes(statevector, relevant_bits=None):
    num = 0
    # Get the number of qubits
    if relevant_bits:
        num_bits = relevant_bits
    else:
        num_bits = statevector.num_qubits
    num_qubits = statevector.num_qubits

    # Iterate through each basis state
    for index, amplitude in enumerate(statevector):
        # Format the basis state as a binary string with leading zeros
        basis_state = format(index, f'0{num_qubits}b')
        relevant_bits = basis_state[-num_bits:]
        flag = check_disagree_list_general(relevant_bits, disagree_list)

        if flag and amplitude != 0:
            num += 1
            print(f"|{relevant_bits}>: {amplitude}")
    print(num)

def print_negative_amplitudes(statevector):
    # Get the number of qubits
    num_qubits = statevector.num_qubits

    # Iterate through each basis state
    for index, amplitude in enumerate(statevector):
        if amplitude.real < 0:
            # Format the basis state as a binary string with leading zeros
            basis_state = format(index, f'0{num_qubits}b')
            print(f"|{basis_state}>: {amplitude}")


class MyTestCase(unittest.TestCase):
    def test_graph_coloring_oracle(self):
        #prep = graph_color_prep(variable_qubits)
        prep = QuantumCircuit(output_qubit + 1)
        prep.h(variable_qubits)
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)
        combined_circuit = QuantumCircuit(output_qubit + 1)
        combined_circuit.compose(prep, inplace=True)
        combined_circuit.compose(oracle, inplace=True)
        state_vector = Statevector(combined_circuit)
        state = Statevector(prep)
        #print_amplitudes(state)
        #print(oracle)
        # Function to filter and print states with negative real part amplitudes
        state = state.evolve(oracle)
        print_amplitudes(state, relevant_bits=8)
        # Call the function with the statevector
        #print_amplitudes(state)
        #self.assertEqual(True, False)  # add assertion here

    def test_graph_triangle_oracle(self):
        tri = TriangleFinding(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/graph_cases/Gset/G3")
        n_nodes = tri.num_nodes
        N = math.comb(n_nodes, 3)
        nodes_qubits = QuantumRegister(n_nodes, name='nodes')
        edge_anc = QuantumRegister(2, name='edge_anc')
        ancilla = QuantumRegister(n_nodes - 2, name='cccx_diff_anc')
        neg_base = QuantumRegister(1, name='check_qubits')
        prep = QuantumCircuit(n_nodes+2+n_nodes-2+1)
        prep.h(tri.graph().nodes)
        oracle = triangle_oracle(tri.edges,nodes_qubits, edge_anc, ancilla, neg_base)
        state = Statevector(prep)
        state = state.evolve(oracle)
        print(oracle)

    def test_equality_checker(self):
        qc = create_equality_checker_circuit(2)
        prep_qc = QuantumCircuit(3 * 2 + 1)
        prep_qc.h([0, 1, 2, 3])
        state = Statevector(prep_qc)
        print_amplitudes(state)
        state = state.evolve(qc)
        print_amplitudes(state)

    def test_graph_coloring(self):
        # CREATE THE PREP AND ORACLE CIRCUITS
        prep = graph_color_prep(variable_qubits)
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)

        # DEFINE THE AmplificationProblem
        def check_disagreement(state): return check_disagree_list_general(state, disagree_list)

        for i in range(100):
            problem = AmplificationProblem(oracle,
                                           state_preparation=prep,
                                           objective_qubits=variable_qubits,
                                           is_good_state=check_disagreement
                                           )

            grover = Grover(iterations=3, sampler=Sampler())
            num = grover.optimal_num_iterations(6, 6)
            results = grover.amplify(problem)
            self.assertEqual(results.oracle_evaluation, True)

    def test_coloring_graph_init(self):
        return


if __name__ == '__main__':
    unittest.main()
