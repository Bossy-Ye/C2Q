import base64
from io import BytesIO

import networkx as nx
import urllib.request
import networkx as nx
import numpy as np
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit import qasm3
from qiskit_algorithms.utils import algorithm_globals
import matplotlib.pyplot as plt


def create_graph_from_edges(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def read_gset_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    num_nodes, num_edges = map(int, lines[0].split())
    elist = []
    for line in lines[1:]:
        if line.strip():
            u, v, weight = map(int, line.split())
            elist.append((u, v, float(weight)))  # Convert to 0-based index and float weight
    return elist, num_nodes, num_edges


def draw_graph(G, colors=None, pos=None, special_nodes=None, transmission=False):
    if colors is None:
        colors = 'lightblue'
    if pos is None:
        pos = nx.spring_layout(G)

        # Initialize node colors
    node_colors = [colors] * G.number_of_nodes()

    special_colors = ['red', 'green', 'blue', 'orange', 'purple']
    # Update colors for special nodes if specified by array
    if special_nodes is not None:
        if all(isinstance(sn, list) for sn in special_nodes):  # If special_nodes is a list of lists
            for group_index, node_group in enumerate(special_nodes):
                color = special_colors[group_index % len(special_colors)]
                for node in node_group:
                    node_colors[node] = color
        else:  # If special_nodes is a single list
            color = 'red'  # Single color for all nodes in the list
            for node in special_nodes:
                node_colors[node] = color

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, node_color=node_colors, pos=pos)

    # Draw edge labels if they exist
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    if not transmission:
        plt.show()
    # Convert graph to base64 string


def get_weight_matrix(G, n):
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    return w
