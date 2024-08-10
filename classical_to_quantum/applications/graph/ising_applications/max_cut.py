from classical_to_quantum.applications.graph.gset import *
from classical_to_quantum.applications.graph.ising_applications.Ising import Ising


class MaxCut(Ising):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.problem = Maxcut(self.w())
        self.qp = self.problem.to_quadratic_program()
        self.qubitOp, offset = self.qp.to_ising()
    
    def plot_res(self, transmission=False):
        super().plot_res(transmission)

# maxcut = MaxCut('/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G7')
# maxcut.run(verbose=True)
# maxcut.show_results()
# maxcut.run_search_parameters()
# maxcut.show_search_results()
# maxcut.plot_search_res()
# draw_graph(maxcut.graph(), transmission=True)
# maxcut.plot_res()
# print(maxcut.generate_qasm3())