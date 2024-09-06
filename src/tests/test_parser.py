from Framework.generator import QASMGenerator

triangle_finding_code = """
import networkx as nx
import itertools

def find_triangles(G):
    triangles = []
    nodes = list(G.nodes)

    # Iterate through all combinations of 3 nodes
    for u, v, w in itertools.combinations(nodes, 3):
        # Check if these nodes form a triangle
        if G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(w, u):
            triangles.append((u, v, w))

    return triangles

# Example usage
G = nx.Graph()

# Add edges to the graph (example graph)
G.add_edges_from([
    (0, 1), (1, 2), (2, 0),  # Triangle between nodes 0, 1, 2
    (0, 3), (3, 4),          # No triangle here
    (1, 4), (2, 3),          # Additional connections
])

# Find all triangles in the graph
triangles = find_triangles(G)

print("Triangles found in the graph:")
for triangle in triangles:
    print(triangle)

# Plot the graph and highlight triangles
def plot_graph_with_triangles(G, triangles):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=15, font_color='black', edge_color='gray')

    # Highlight the triangles
    for triangle in triangles:
        nx.draw_networkx_edges(G, pos, edgelist=[(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])],
                               width=8, alpha=0.5, edge_color='green')

    plt.title("Triangles in the Graph")
    plt.show()

# Plot the graph with triangles highlighted
plot_graph_with_triangles(G, triangles)
"""
generator = QASMGenerator()
generator.qasm_generate(triangle_finding_code, verbose=True)

