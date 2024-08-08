import matplotlib.pyplot as plt
import networkx as nx
import os

import numpy as np

from classical_to_quantum.applications.graph.optimization_solver import *
from classical_to_quantum.applications.graph.gset import *
from classical_to_quantum.applications.tools import OptimizerLog
from qiskit.circuit import QuantumCircuit
import networkx as nx
import pylab


class GraphProblem:
    def __init__(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path
        self.elist, self.num_nodes, self.num_edges = read_gset_file(file_path)
        self._graph = create_graph_from_edges(self.elist)
        self._w = get_weight_matrix(self._graph, self.num_nodes)
        self.nodes_results = None

    def run(self):
        raise NotImplementedError("not implemented yet")

    def generate_qasm3(self):
        raise NotImplementedError("not implemented yet")

    def plot_res(self, transmission=False):
        """children class should implement this function to plot convergence convex
        or picked nodes/edges in original graph"""
        #
        draw_graph(self._graph, special_nodes=self.nodes_results, transmission=transmission)

    def show_results(self):
        #print intermediate procedures
        raise NotImplementedError("not implemented yet")

    def graph(self) -> nx.Graph:
        return self._graph

    def w(self):
        return self._w
