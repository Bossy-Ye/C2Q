from typing import Dict, Union, List, Set

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, Clbit, Measure
from qiskit.result import Result


def parse_result(
        qiskit_result: Result, circuit: QuantumCircuit, measurement_names: Set[str]={Measure().name},
        indexed_results: bool = True) -> Dict[Union[Qubit, int], Dict[str, int]]:
    """
    parse results into a dictionary where keys are the qubits (or its indices) and values are dictionaries containing
    results of states for that qubit (ignoring all other qubits)
    :param qiskit_result: result returned by backend.run(circuit)
    :param circuit: circuit for which the qiskit_result was run
    :param indexed_results: if true keys for dictionary are indices if false it's Qubit objects
    :return: dictionary containing parsed results
    """
    qubit_clbit_mapping = _get_qubit_mapping(circuit, measurement_names)
    parsed_results = {}
    for state, count in qiskit_result.get_counts().items():
        parsed_state = _parse_state(state)
        for qubit_index in range(len(circuit.qubits)):
            key = qubit_index if indexed_results else circuit.qubits[qubit_index]
            measurement_index = qubit_clbit_mapping[qubit_index]

            if measurement_index is None:
                continue

            qubit_state = parsed_state[measurement_index]

            if key not in parsed_results:
                parsed_results[key] = {'0': 0, '1': 0}
            parsed_results[key][qubit_state] += count

    return parsed_results


def _get_qubit_mapping(circuit: QuantumCircuit, measurement_names: Set[str]) -> List[Union[None, int]]:
    """
    return mapping between index of a qubit and a real index of its
    measurement result in the state returned by get_counts(). indices do not consider spaces so state must have spaces
    removed before indices address correctly
    :param circuit: circuit for which to create the mapping
    :return: mapping between qubit indices and its measurement indices
    """
    qubit_mapping = [None] * circuit.num_qubits
    for instruction, qubits, bits in circuit.data:
        if instruction.name in measurement_names:
            qubit_index = circuit.qubits.index(qubits[0])
            qubit_mapping[qubit_index] = _get_real_clbit_index(circuit, bits[0])
    return qubit_mapping


def _get_real_clbit_index(circuit: QuantumCircuit, clbit: Clbit) -> int:
    """
    return the index of classical bit in the state returned by counts (indices don't consider spaces
    - they must be removed for the indices to address correctly)
    :param circuit: circuit that clbit is part of
    :param clbit: clbit to get index of
    :return:
    """
    creg_index = 0
    for creg in reversed(circuit.cregs):
        if clbit in creg:
            clbit_index = creg.index(clbit)
            return creg_index + (len(creg) - clbit_index - 1)
        else:
            creg_index += len(creg)


def _parse_state(state: str) -> str:
    """
    removes spaces from str
    :param state: str to parse
    :return: provided str without spaces
    """
    return state.replace(" ", "")
