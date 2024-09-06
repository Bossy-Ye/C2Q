import numpy as np
from qiskit.circuit.library import CDKMRippleCarryAdder, RGQFTMultiplier
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.library import VBERippleCarryAdder
import pennylane as qml
import matplotlib.pyplot as plt
from src.utils import *


def add_k_fourier(k, wires):
    for j in range(len(wires)):
        qml.RZ(k * np.pi / 2 ** j, wires=wires[j])


def quantum_mul_pennylane(m, k, verbose=False):
    wires_m = list(range(minimum_bits_required(m)))
    wires_k = list(range(len(wires_m), minimum_bits_required(k) + len(wires_m)))
    # m and k codification
    wires_solution = list(
        range(len(wires_m) + len(wires_k), len(wires_m) + len(wires_k) + max(len(wires_m), len(wires_k)) * 2))
    dev = qml.device("default.qubit", wires=wires_m + wires_k + wires_solution, shots=1)

    @qml.qnode(dev)
    def mul(m, k):
        qml.BasisEmbedding(m, wires=wires_m)
        qml.BasisEmbedding(k, wires=wires_k)

        # Apply multiplication
        multiplication_pennylane(wires_m, wires_k, wires_solution)

        return qml.sample(wires=wires_solution)

    if verbose:
        qml.draw_mpl(mul, show_all_wires=True)(m, k)
        plt.show()
    return mul(m, k)


def multiplication_pennylane(wires_m, wires_k, wires_solution):
    # prepare sol-qubits to counting
    qml.QFT(wires=wires_solution)

    # add m to the counter
    for i in range(len(wires_k)):
        for j in range(len(wires_m)):
            coeff = 2 ** (len(wires_m) + len(wires_k) - i - j - 2)
            qml.ctrl(add_k_fourier, control=[wires_k[i], wires_m[j]])(coeff, wires_solution)

    # return to computational basis
    qml.adjoint(qml.QFT)(wires=wires_solution)


def quantum_multiplication(A, B, verbose=False):
    # Determine the number of qubits required
    num_state_qubits = max(A.bit_length(), B.bit_length())
    num_result_qubits = num_state_qubits * 2

    # Create Quantum and Classical Registers
    q = QuantumRegister(num_state_qubits * 2 + num_result_qubits, 'q')
    c = ClassicalRegister(num_result_qubits, 'c')
    circuit = QuantumCircuit(q, c)

    # Set the bits for Operand A (from right to left)
    for i in range(A.bit_length()):
        if (A >> i) & 1:
            circuit.x(q[i])

    # Set the bits for Operand B (from right to left)
    for i in range(B.bit_length()):
        if (B >> i) & 1:
            circuit.x(q[i + num_state_qubits])

    # Create the RGQFTMultiplier circuit and compose it with the main circuit
    multiplier_circuit = RGQFTMultiplier(num_state_qubits=num_state_qubits, num_result_qubits=num_result_qubits)
    circuit = circuit.compose(multiplier_circuit)

    # Measure the result
    for i in range(num_result_qubits):
        circuit.measure(q[i + num_state_qubits * 2], c[i])

    # Print the circuit for verification
    if verbose: print(circuit)

    # Run the circuit using the Sampler
    sampler = Sampler()
    result = sampler.run(circuit).result()
    # Extract the result from the Sampler output
    result_counts = result.quasi_dists[0]
    if verbose: print(f'sampling results: {result_counts}')
    result_value = max(result_counts, key=result_counts.get)

    return result_value, circuit


def decimal_to_binary_list(num, n_bits):
    return [int(x) for x in bin(num)[2:].zfill(n_bits)][::-1]


def decimal_to_complement_binary_list(num, n_bits):
    if num >= 0:
        # Positive number, convert to binary and pad to n_bits
        bin_str = bin(num)[2:]  # Remove '0b' prefix
        if len(bin_str) > n_bits:
            raise ValueError("Number does not fit in the specified number of bits")
        bin_str = bin_str.zfill(n_bits)  # Pad with leading zeros
    else:
        # Negative number, calculate two's complement
        num = (1 << n_bits) + num  # Equivalent to (2^n_bits) + num
        bin_str = bin(num)[2:]  # Remove '0b' prefix

        # Convert binary string to list of integers (0s and 1s)
    bin_list = [int(bit) for bit in bin_str]

    # Reverse the list for little-endian representation
    little_endian_list = bin_list[::-1]
    return little_endian_list


def complement_binary_list_to_decimal(bin_list):
    # Reverse the little-endian list to get the big-endian list
    bin_list = bin_list[::-1]

    # Determine if the number is negative (MSB is 1)
    is_negative = bin_list[0] == 1

    # Convert the big-endian binary list to a string
    bin_str = ''.join(str(bit) for bit in bin_list)

    if is_negative:
        # Convert from two's complement
        # Invert the bits
        inverted_bin_list = [1 - bit for bit in bin_list]
        # Convert the inverted list to a string
        inverted_bin_str = ''.join(str(bit) for bit in inverted_bin_list)
        # Convert the inverted binary string to a decimal number and add 1
        decimal_value = int(inverted_bin_str, 2) + 1
        # Negate the result to get the original negative number
        decimal_value = -decimal_value
    else:
        # Directly convert the binary string to a decimal number
        decimal_value = int(bin_str, 2)

    return decimal_value


def quantum_add(left, right):
    n_bits = max(left.bit_length(), right.bit_length())+1
    # Ensure both left and right have n_bits length
    left_list = decimal_to_complement_binary_list(left, n_bits)
    right_list = decimal_to_complement_binary_list(right, n_bits)

    # Create a quantum circuit with 2*n_bits for input and 1 additional for carry
    if left * right > 0:
        qc = QuantumCircuit(3 * n_bits + 1, n_bits + 1)
    else:
        qc = QuantumCircuit(3 * n_bits + 1, n_bits)
    # Initialize the input qubits
    for i in range(n_bits):
        if left_list[i] == 1:
            qc.x(i)  # Set qubit for left bit
        if right_list[i] == 1:
            qc.x(n_bits + i)  # Set qubit for right bit

    # Apply quantum gates for addition
    for i in range(n_bits):
        qc.ccx(i, n_bits + i, 2 * n_bits + i + 1)
        qc.cx(i, n_bits + i)
        qc.ccx(n_bits + i, 2 * n_bits + i, 2 * n_bits + i + 1)
        qc.cx(n_bits + i, 2 * n_bits + i)
        qc.cx(i, n_bits + i)

    # Measuring the result
    for i in range(n_bits):
        qc.measure(i + 2 * n_bits, i)  # Measure left bits to result bits
    if left * right > 0:
        qc.measure(3 * n_bits, n_bits)
    sampler = Sampler()
    result = sampler.run(circuits=qc, shots=1024).result()
    result = list(result.quasi_dists[0].keys())[0]
    if left * right > 0:
        result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
    else:
        result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
    return result, qc


def quantum_subtract(left, right):
    n_bits = max(left.bit_length(), right.bit_length())
    # Ensure both minuend and subtrahend have n_bits length
    minuend = decimal_to_complement_binary_list(left, n_bits)
    subtrahend = decimal_to_complement_binary_list(right, n_bits)

    # Create a quantum circuit with 2*n_bits for input and 1 additional for borrow
    adder = VBERippleCarryAdder(num_state_qubits=n_bits)
    num_qubits = len(adder.qubits)
    if left * right < 0:
        qc = QuantumCircuit(num_qubits, n_bits + 1)
    else:
        qc = QuantumCircuit(num_qubits, n_bits)
    # Initialize the input qubits
    for i in range(n_bits):
        if minuend[i] == 1:
            qc.x(i + 1)  # Set qubit for minuend bit
        if subtrahend[i] == 1:
            qc.x(i + 1 + n_bits)  # Set qubit for subtrahend bit

    qc.barrier()
    qc.x(range(n_bits + 1, 2 * n_bits + 1))

    qc.x(0)

    qc.append(adder, range(num_qubits))
    for i in range(n_bits):
        qc.measure(i + n_bits + 1, i)
    if left * right < 0:
        qc.measure(n_bits + n_bits + 1, n_bits)
    sampler = Sampler()
    result = sampler.run(circuits=qc, shots=1024).result()
    result = list(result.quasi_dists[0].keys())[0]
    if left * right < 0:
        result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
    else:
        result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
    return result, qc


# qc, res = quantum_subtract(5, -16, 5)
# # qc = quantum_subtract(23, 6, 5)
# # print(qc)
# # sampler = Sampler()
# # result = sampler.run(qc, shots=1024).result()
# print(qc)
# print(res)

def permutations(list):
    result = []

    def f(start):
        if start == len(list):
            result.append(list[:])
        for i in range(start, len(list)):
            list[start], list[i] = list[i], list[start]
            f(i + 1)
            list[start], list[i] = list[i], list[start]

    f(0)
    return result

