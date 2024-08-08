import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def qubit_num(n):
    """
    Calculate the function f(n) defined as the base-2 logarithm of n using numpy.

    Parameters:
    n (int): A positive integer that is a power of 2.

    Returns:
    int: The base-2 logarithm of n.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2.")

    return int(np.log2(n))


def generate_pauli_operators(n):
    """
    Generate all Pauli operators for n qubits.

    Parameters:
    n (int): Number of qubits (dimension of the matrix is 2^n x 2^n).

    Returns:
    list: List of Pauli strings for n qubits.
    """
    pauli_strings = []
    num_paulis = 4 ** n

    for i in range(num_paulis):
        pauli_str = ''
        temp = i
        for _ in range(n):
            if temp % 4 == 0:
                pauli_str = 'I' + pauli_str
            elif temp % 4 == 1:
                pauli_str = 'X' + pauli_str
            elif temp % 4 == 2:
                pauli_str = 'Y' + pauli_str
            elif temp % 4 == 3:
                pauli_str = 'Z' + pauli_str
            temp //= 4
        pauli_strings.append(pauli_str)

    return pauli_strings


def decompose_into_pauli(matrix):
    """
    Decompose a square matrix into a linear combination of Pauli matrices using SparsePauliOp,
    ignoring the imaginary part of the matrix.

    Parameters:
    matrix (numpy.ndarray): A square matrix with dimensions 2^k x 2^k.

    Returns:
    SparsePauliOp: A SparsePauliOp object containing the linear combination of Pauli operators.
    """
    dim = matrix.shape[0]
    if not (np.log2(dim) % 1 == 0):
        raise ValueError("Matrix dimension must be a power of 2.")

    n = int(np.log2(dim))

    # Generate Pauli operators for n qubits
    pauli_strings = generate_pauli_operators(n)
    pauli_ops = [SparsePauliOp.from_list([(ps, 1)]).to_matrix() for ps in pauli_strings]

    # Compute coefficients
    coeffs = []
    for pauli_matrix in pauli_ops:
        coeff = np.trace(np.dot(matrix, pauli_matrix)) / (2 ** n)
        if not np.isclose(np.imag(coeff), 0):
            raise ValueError("Imaginary part of matrix is nonzero... Other words, the matrix provided is not "
                             "representable by pauli matrix, maybe try another one??")
        coeffs.append(coeff)
    # Create SparsePauliOp with the coefficients
    pauli_op = SparsePauliOp.from_list([(ps, coeff) for ps, coeff in zip(pauli_strings, coeffs)])

    return pauli_op


def decompose_into_pauli1(matrix):
    """
    Decompose a 2x2 matrix into a linear combination of Pauli matrices.

    Parameters:
    matrix (numpy.ndarray): A 2x2 complex matrix.

    Returns:
    tuple: Coefficients (alpha_0, alpha_x, alpha_y, alpha_z) for the linear combination.
    """
    # Define the Pauli matrices
    sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Calculate the coefficients
    alpha_0 = 0.5 * np.trace(np.dot(matrix, sigma_0))
    alpha_x = 0.5 * np.trace(np.dot(matrix, sigma_x))
    alpha_y = 0.5 * np.trace(np.dot(matrix, sigma_y))
    alpha_z = 0.5 * np.trace(np.dot(matrix, sigma_z))

    return SparsePauliOp.from_list([
        ('I', alpha_0),
        ('X', alpha_x),
        ('Y', alpha_y),
        ('Z', alpha_z)
    ])

def get_evolved_state(circuit: QuantumCircuit, statevector: Statevector, verbose=False):
    final_statevector = statevector.evolve(circuit)
    amplitudes = final_statevector.probabilities_dict()
    # To see the amplitudes in a readable format:
    if verbose:
        print("Amplitudes for qubit nodes:")
        for state, probability in amplitudes.items():
            amplitude = final_statevector[state]
            print(f"{state}: {amplitude}")
            print(f"Probability: {probability}")
    return final_statevector, amplitudes
