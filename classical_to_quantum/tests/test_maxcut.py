from applications.graph.ising_applications.max_cut import MaxCut

max_cut = MaxCut("/Users/mac/workspace/quantum-journey2/classical_to_quantum/graph_cases/Gset/G2")
max_cut.run(verbose=True)
max_cut.show_results()
qasm = max_cut.generate_qasm3()
print(qasm)