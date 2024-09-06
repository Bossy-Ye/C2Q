import networkx as nx
from matplotlib import pyplot as plt


class Interpreter:
    @staticmethod
    def draw_tsp_solution(G:nx.Graph, order):
        colors = colors = ["r" for node in G.nodes]
        pos = nx.spring_layout(G)
        G2 = nx.DiGraph()
        G2.add_nodes_from(G)
        n = len(order)
        for i in range(n):
            j = (i + 1) % n
            G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
        )
        edge_labels = nx.get_edge_attributes(G2, "weight")
        nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)