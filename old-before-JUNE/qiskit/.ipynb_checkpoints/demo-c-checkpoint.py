from qiskit.circuit.library import HGate
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.append(
  HGate(),
  [0]
)

qc.draw("mpl");
