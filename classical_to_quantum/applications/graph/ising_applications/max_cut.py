import qiskit.qasm2
from openqaoa import create_device

from classical_to_quantum.applications.graph.gset import *
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising
from openqaoa.problems import MaximumCut
from openqaoa import QAOA

class MaxCut(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = MaximumCut(self.graph())
        self.maxcut_qubo = self.problem.qubo
        qaoa = QAOA()

        # optionally configure the following properties of the model

        # device
        qiskit_device = create_device(location='local', name='qiskit.statevector_simulator')
        qaoa.set_device(qiskit_device)

        # circuit properties
        qaoa.set_circuit_properties(p=2, param_type='standard', init_type='rand', mixer_hamiltonian='x')

        # backend properties (already set by default)
        qaoa.set_backend_properties(prepend_state=None, append_state=None)

        # classical optimizer properties
        qaoa.set_classical_optimizer(method='nelder-mead', maxiter=200, tol=0.001,
                                     optimization_progress=True, cost_progress=True, parameter_log=True)

        self.qaoa = qaoa

    def run(self, verbose=False):
        self.qaoa.compile(self.maxcut_qubo)
        self.qaoa.optimize()

        self.is_executed = True
        result = self.qaoa.result
        if verbose:
            print(result.optimized)
        return result

    def generate_qasm(self):
        variational_params = self.qaoa.optimizer.variational_params
        optimized_angles = self.qaoa.result.optimized['angles']
        variational_params.update_from_raw(optimized_angles)
        optimized_circuit = self.qaoa.backend.qaoa_circuit(variational_params)
        return qiskit.qasm2.dumps(optimized_circuit)
    @staticmethod
    def random_instance(self):
        return MaximumCut.random_instance()

    def plot_res(self, transmission=False):
        super().plot_res(transmission)


maxcut = MaxCut('/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/graph_cases/Gset/G0')
maxcut.run(verbose=True)
# maxcut.show_results()
# maxcut.run_search_parameters()
# maxcut.show_search_results()
# maxcut.plot_search_res()
# draw_graph(maxcut.graph(), transmission=True)
# maxcut.plot_res()
print(maxcut.generate_qasm())
