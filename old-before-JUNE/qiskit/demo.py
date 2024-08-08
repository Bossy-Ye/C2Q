import numpy as np
import qiskit.qasm3

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

pm = generate_preset_pass_manager()
backend = AerSimulator()
# Step 1: Map classical inputs to a quantum problem

theta = Parameter("θ")

chsh_circuit = QuantumCircuit(2)
chsh_circuit.h(0)
chsh_circuit.cx(0, 1)
chsh_circuit.ry(theta, 0)

number_of_phases = 21
phases = np.linspace(0, 2 * np.pi, number_of_phases)
individual_phases = [[ph] for ph in phases]

ZZ = SparsePauliOp.from_list([("ZZ", 1)])
ZX = SparsePauliOp.from_list([("ZX", 1)])
XZ = SparsePauliOp.from_list([("XZ", 1)])
XX = SparsePauliOp.from_list([("XX", 1)])
ops = [ZZ, ZX, XZ, XX]



# Step 2: Optimize problem for quantum execution.

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
chsh_isa_circuit = pm.run(chsh_circuit)
isa_observables = [operator.apply_layout(chsh_isa_circuit.layout) for operator in ops]

from qiskit_ibm_runtime import EstimatorV2 as Estimator

# Step 3: Execute using Qiskit primitives.

# Reshape observable array for broadcasting
reshaped_ops = np.fromiter(isa_observables, dtype=object)
reshaped_ops = reshaped_ops.reshape((4, 1))

estimator = Estimator(backend, options={"default_shots": int(1e4)})
job = estimator.run([(chsh_isa_circuit, reshaped_ops, individual_phases)])
# Get results for the first (and only) PUB
pub_result = job.result()[0]
print(f">>> Expectation values: {pub_result.data.evs}")
print(f">>> Standard errors: {pub_result.data.stds}")
print(f">>> Metadata: {pub_result.metadata}")