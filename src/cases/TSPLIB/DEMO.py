import tsplib95
import networkx as nx
import numpy as np


def create_tsplib_graph():
    # Create a sample graph
    G = nx.complete_graph(5)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = 1 + (u + v) % 5  # Sample weights for demonstration

    return G


def graph_to_tsplib(G):
    # Initialize a TSPLIB problem
    problem = tsplib95.models.StandardProblem()
    # Set the problem attributes
    problem.name = "SampleTSP"
    problem.type = "TSP"
    problem.dimension = G.number_of_nodes()
    problem.edge_weight_type = "EXPLICIT"
    problem.edge_weight_format = "FULL_MATRIX"

    # Create the edge weight matrix
    edge_weight_matrix = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=int)
    for (u, v, data) in G.edges(data=True):
        edge_weight_matrix[u][v] = data['weight']
        edge_weight_matrix[v][u] = data['weight']  # Since the graph is undirected

    problem.edge_weights = edge_weight_matrix.tolist()

    return problem


def write_tsplib(problem, filename):
    with open(filename, 'w') as f:
        f.write(problem.__str__())


# Create and convert the graph
G = create_tsplib_graph()
problem = graph_to_tsplib(G)

# Write the TSPLIB problem to a file
write_tsplib(problem, 'sample.tsp')
