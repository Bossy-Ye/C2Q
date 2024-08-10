from openqaoa.problems import MaximumCut, MinimumVertexCover
import networkx as nx

g = nx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field =1.0, penalty=10)
qubo_problem = vc.qubo

print(qubo_problem)



