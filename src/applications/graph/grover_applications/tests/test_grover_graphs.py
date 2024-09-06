import unittest

import networkx
import qiskit.qasm2

from src.applications.graph.grover_applications.graph_color import *
from qiskit.quantum_info import Statevector
from src.applications.graph.grover_applications.triangle_finding import *
from src.applications.graph.grover_applications.graph_oracle import *
from utils import *
from src.applications.graph.grover_applications.grover_auxiliary import *


def func(state):
    return True


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


def print_all_amplitudes(statevector):
    # Get the number of qubits
    num_qubits = statevector.num_qubits

    # Iterate through each basis state
    for index, amplitude in enumerate(statevector):
        # Format the basis state as a binary string with leading zeros
        basis_state = format(index, f'0{num_qubits}b')
        print(f"|{basis_state}>: {amplitude}")


variable_qubits = [0, 1, 2, 3, 4, 5, 6, 7]
check_qubits = [8, 9, 10, 11, 12, 13]

disagree_list = [[[0, 1], [2, 3]],
                 [[0, 1], [4, 5]],
                 [[0, 1], [6, 7]],
                 [[2, 3], [4, 5]],
                 [[2, 3], [6, 7]],
                 [[4, 5], [6, 7]]
                 ]
output_qubit = 14


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
        tri = GraphProblem(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/src/cases/Gset/G3")
        n_nodes = tri.num_nodes
        print(tri.elist)
        N = 2 ** n_nodes
        iterations = math.floor(math.pi / 4 * math.sqrt(N))
        nodes_qubits = QuantumRegister(n_nodes, name='nodes')
        edge_anc = QuantumRegister(2, name='edge_anc')
        ancilla = QuantumRegister(n_nodes - 2, name='cccx_diff_anc')
        neg_base = QuantumRegister(1, name='check_qubits')
        sub_qbits = QuantumRegister(n_nodes)
        next_qubits = QuantumRegister(n_nodes + 1)
        prep = QuantumCircuit(sub_qbits, next_qubits, name="state_prep")
        #prep, sub_qbits = wn(prep, sub_qbits)
        #prep.x(sub_qbits)
        prep.h(sub_qbits)
        oracle = triangle_oracle(tri.elist, nodes_qubits, edge_anc, ancilla, neg_base)
        print(prep.num_qubits, oracle.num_qubits)
        grover = GroverWrapper(oracle=oracle,
                               iterations=iterations,
                               state_preparation=prep,
                               is_good_state=func,
                               objective_qubits=list(range(n_nodes)))
        grover.run(verbose=True)
        # state = Statevector(prep)
        # print(oracle.num_qubits, prep.num_qubits)
        # state = state.evolve(oracle)
        # print_all_amplitudes(state)

    def test_equality_checker(self):
        qc = QuantumCircuit(8)
        state = Statevector.from_label('000000000011011')
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)
        get_evolved_state(oracle, state, verbose=True)

    def test_graph_coloring(self):
        # CREATE THE PREP AND ORACLE CIRCUITS
        prep = graph_color_prep(variable_qubits)
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)

        # DEFINE THE AmplificationProblem
        def check_disagreement(state): return check_disagree_list_general(state, disagree_list)

        problem = AmplificationProblem(oracle,
                                       state_preparation=prep,
                                       objective_qubits=variable_qubits,
                                       is_good_state=check_disagreement
                                       )

        grover = Grover(iterations=1, sampler=Sampler())
        results = grover.amplify(problem)
        print(results)
        self.assertEqual(True, True)

    def test_coloring_graph_grover_wrapper(self):
        prep = graph_color_prep(variable_qubits)
        oracle = graph_color_oracle(disagree_list, variable_qubits, check_qubits, output_qubit)

        # DEFINE THE AmplificationProblem
        def check_disagreement(state): return check_disagree_list_general(state, disagree_list)

        grover = GroverWrapper(oracle=oracle,
                               iterations=1,
                               is_good_state=check_disagreement,
                               objective_qubits=variable_qubits)
        grover.run(verbose=True)
        str = grover.export_to_qasm()
        sampler = Sampler()
        circuit = grover.circuit
        res = sampler.run(circuit).result()
        print(str)
        circuit = qiskit.qasm2.loads(str)
        print(circuit)

    def test_clique_grover(self):
        G = networkx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
        print(G.edges)
        cnf = clique_to_sat(G, 2)
        oracle = cnf_to_quantum_oracle_optimized(cnf)
        print(oracle)

        def func(state):
            return True

        prep = QuantumCircuit(cnf.nv)
        prep.h(list(range(cnf.nv)))
        grover = GroverWrapper(oracle=oracle,
                               iterations=2,
                               state_preparation=prep,
                               is_good_state=func,
                               objective_qubits=list(range(cnf.nv)))
        try:
            grover.run(verbose=True)
        except Exception as e:
            print(e)

    def test_triangle_graph_grover(self):
        triangle = TriangleFinding(
            "//cases/Gset/G8")

        result = triangle.run(verbose=True)

        from src.applications.graph.grover_applications.grover_auxiliary import get_top_measurements, \
            plot_triangle_finding
        top_measurements = get_top_measurements(result, 0.001, num=20)
        print(top_measurements)
        plot_triangle_finding(triangle.graph(), top_measurements)
        print(triangle.export_to_qasm())


if __name__ == '__main__':
    unittest.main()
