from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import os

import networkx.classes.graph
import numpy as np

from classical_to_quantum.applications.graph.optimization_solver import *
from classical_to_quantum.applications.graph.gset import *
from classical_to_quantum.applications.tools import OptimizerLog
from qiskit.circuit import QuantumCircuit
import networkx as nx
import pylab
from qiskit import qasm2
from classical_to_quantum.classiq_exceptions import *


class GraphProblem:
    def __init__(self, input_data: str | networkx.classes.graph.Graph):
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(input_data)
            self.file_path = input_data
            try:
                self.elist, self.num_nodes, self.num_edges = read_gset_file(input_data)
            except Exception as e:
                raise FileLoadingError(input_data, message=str(e))
            self._graph = create_graph_from_edges(self.elist)
        elif isinstance(input_data, networkx.classes.graph.Graph):
            self._graph = input_data
            self.num_nodes = self._graph.number_of_nodes()
            self.num_edges = self._graph.number_of_edges()
            self.elist = self._graph.edges
        else:
            raise InvalidInputError(type(input_data))

    def run(self):
        pass
        #raise NotImplementedError("not implemented yet")

    def run_on_quantum(self):
        raise NotImplementedError("not implemented yet")

    def generate_qasm(self):
        raise NotImplementedError("not implemented yet")

    def plot_graph_solution(self):
        """children class should implement this function to plot convergence convex
        or picked nodes/edges in original graph"""
        raise NotImplementedError("not implemented yet")
        #draw_graph(self._graph, special_nodes=self.nodes_results, transmission=transmission)

    def plot_graph(self):
        """
        plot original graph
        Returns
        -------

        """
        raise NotImplementedError("not implemented yet")

    def plot_results(self):
        """
        intermediate results
        Returns
        -------

        """
        raise NotImplementedError("not implemented yet")

    def graph(self) -> nx.Graph:
        return self._graph
