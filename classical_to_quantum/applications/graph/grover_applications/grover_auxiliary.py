from matplotlib import pyplot as plt
from qiskit_algorithms import *
from typing import List, Dict
import networkx as nx

def get_top_measurements(result: GroverResult, threshold: float = 0.001, num: int = 3) -> List[Dict[str, float]]:
    """
    Get the top `num` measurements from the circuit results that differ within a given threshold.

    Parameters:
    - result: An object containing the assignment and circuit_results.
    - threshold (float): The threshold within which the measurement probabilities should differ.
    - num (int): The number of top measurements to return.

    Returns:
    - List[Dict[str, float]]: A list of dictionaries containing the top measurements that meet the threshold criteria.
    """
    # Extract the circuit results from the result object
    circuit_results = result.circuit_results

    # Find the highest probability in the circuit results
    max_prob = max(circuit_results[0].values())

    # Filter measurements that are within the threshold of the maximum probability
    filtered_measurements = []
    for measurement, probability in circuit_results[0].items():
        if abs(max_prob - probability) <= threshold:
            filtered_measurements.append({measurement: probability})

    # Sort the filtered measurements by probability in descending order
    filtered_measurements.sort(key=lambda x: list(x.values())[0], reverse=True)

    # Return only the top `num` measurements
    return filtered_measurements[:num]


def plot_graph_independent_set(G, is_assignment, ax):
    """
    Plot the graph highlighting the independent set based on the given assignment.

    Parameters:
    - G (networkx.Graph): The graph to be plotted.
    - is_assignment (str): A bitstring where '1' indicates a node is in the independent set and '0' indicates it is not.
    - ax (matplotlib.axes.Axes): The axis on which to plot the graph.
    """
    # Decode the independent set assignment
    independent_set = []
    for i, bit in enumerate(reversed(is_assignment)):  # Start from the right-hand side of the bitstring
        if bit == '1':
            independent_set.append(i)

    # Color nodes: red for nodes in the independent set, lightgray for others
    node_colors = ['red' if node in independent_set else 'lightgray' for node in G.nodes]

    # Plot the graph on the given axis
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(G, pos, node_color=node_colors, with_labels=True, edge_color='black', node_size=500, font_weight='bold',
            ax=ax)

    # Highlight the nodes that are in the independent set
    nx.draw_networkx_nodes(G, pos, nodelist=independent_set, node_color='red', ax=ax)

    ax.set_title(f"Independent Set: {is_assignment}")


def plot_multiple_independent_sets(G, measurements, num_per_row=3):
    """
    Plot multiple independent set solutions in a grid layout.

    Parameters:
    - G (networkx.Graph): The graph to be plotted.
    - measurements (list of dict): A list of dictionaries containing bitstrings as keys.
    - num_per_row (int): The number of plots per row in the grid.
    """
    num_measurements = len(measurements)
    num_rows = (num_measurements + num_per_row - 1) // num_per_row  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_per_row, figsize=(5 * num_per_row, 5 * num_rows))
    axs = axs.flatten()  # Flatten the axes array for easy indexing

    for i, measurement in enumerate(measurements):
        is_assignment = list(measurement.keys())[0]  # Get the bitstring
        plot_graph_independent_set(G, is_assignment, ax=axs[i])

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


# Mapping from bit pairs to colors
color_map = {
    '00': 'red',  # Color 0
    '01': 'green',  # Color 1
    '10': 'blue',  # Color 2
    '11': 'yellow'  # Color 3
}


def plot_graph_coloring(G, color_assignment, ax):
    """
    Plot the graph coloring using the provided color assignment.

    Parameters:
    - G (networkx.Graph): The graph to be colored.
    - color_assignment (dict): A dictionary mapping node indices to colors.
    - ax (matplotlib.axes.Axes): The axis on which to plot the graph.
    """
    # Generate the node colors based on the assignment
    node_colors = [color_assignment[node] for node in G.nodes]

    # Draw the graph with the assigned colors
    pos = nx.circular_layout(G)  # You can change the layout as needed
    nx.draw(G, pos, ax=ax, node_color=node_colors, with_labels=True, edge_color='black', node_size=500,
            font_weight='bold')
    ax.set_title("Graph Coloring")


def decode_bitstring_to_colors(bitstring):
    """
    Decode the bitstring into a color assignment based on the last two bits.

    Parameters:
    - bitstring (str): The bitstring representing the color assignment.

    Returns:
    - dict: A dictionary mapping node indices to color names.
    """
    n = len(bitstring) // 2  # Number of nodes
    color_assignment = {}

    for i in range(n):
        color_bits = bitstring[2 * i:2 * i + 2]  # Extract the bit pair for the i-th node
        color_assignment[i] = color_map[color_bits]  # Map the bit pair to the corresponding color

    return color_assignment


# Plotting multiple graph colorings
def plot_multiple_graph_colorings(G, bitstring_results, num_per_row=3):
    """
    Plot multiple graph colorings in a grid layout.

    Parameters:
    - G (networkx.Graph): The graph to be colored.
    - bitstring_results (list): A list of dictionaries representing color assignments.
    - num_per_row (int): Number of plots per row.
    """
    num_colorings = len(bitstring_results)
    num_rows = (num_colorings + num_per_row - 1) // num_per_row  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_per_row, figsize=(15, 5 * num_rows))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten the axis array for easy indexing

    for i, result in enumerate(bitstring_results):
        bitstring = list(result.keys())[0]
        color_assignment = decode_bitstring_to_colors(bitstring)
        print(f"Plotting for bitstring {bitstring} with color assignment: {color_assignment}")
        plot_graph_coloring(G, color_assignment, axs[i])

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()