import math

from classical_to_quantum.applications.arithmetic.quantum_arithmetic import *


def quantum_factor_mul_oracle(n):
    num_result_qubits = n.bit_length()
    num_state_qubits = math.ceil(num_result_qubits / 2)

    obj_bits = list(range(0, num_state_qubits))

    # Create Quantum and Classical Registers, one more ancilla bit
    q = QuantumRegister(num_state_qubits * 2 + num_result_qubits + 1, 'q')
    #c = ClassicalRegister(num_result_qubits + 1, 'c')
    circuit = QuantumCircuit(q)
    prep_state = QuantumCircuit(q)

    prep_state.h(list(range(num_state_qubits * 2)))

    multiplier_circuit = RGQFTMultiplier(num_state_qubits=num_state_qubits, num_result_qubits=num_result_qubits)
    # super position
    #circuit.h(list(range(num_state_qubits * 2)))

    # flip bit
    circuit.x(q[circuit.num_qubits - 1])
    circuit.h(q[circuit.num_qubits - 1])

    # compose
    circuit = circuit.compose(multiplier_circuit)

    binary_n = format(n, f'0{num_result_qubits}b')
    # Flip the 0s
    for i in range(num_result_qubits):
        if binary_n[num_result_qubits - i - 1] == '0':
            circuit.x(q[i + num_state_qubits * 2])

    # Apply multi-controlled Z (or CNOT) to flip the ancilla qubit if the result matches `n`
    circuit.mcx(list(range(num_state_qubits * 2, num_state_qubits * 2 + num_result_qubits)),
                q[circuit.num_qubits - 1])

    # Uncompute the X gates applied earlier
    for i in range(num_result_qubits):
        if binary_n[num_result_qubits - i - 1] == '0':
            circuit.x(q[i + num_state_qubits * 2])

    # Uncompute mul
    circuit = circuit.compose(multiplier_circuit.inverse())
    return circuit, prep_state, obj_bits

