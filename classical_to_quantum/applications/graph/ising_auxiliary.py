import networkx
import numpy as np
from matplotlib import pyplot as plt


def plot_first_valid_coloring_solutions(solutions, coloring_problem_ising):
    try:
        # Calculate the number of rows needed (up to 20 solutions, 3 per row)
        valid_solution_count = 0
        max_solutions = 3
        cols = 3
        rows = (max_solutions + cols - 1) // cols  # Ceiling division to determine the number of rows

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust figure size based on number of rows

        # Flatten axs array in case of multiple rows/columns
        axs = axs.flatten()

        for i, solution in enumerate(solutions):
            if valid_solution_count >= max_solutions:
                break  # Stop after plotting 20 valid solutions

            try:
                ax = axs[valid_solution_count]  # Access the correct subplot
                # Plot each valid solution
                coloring_problem_ising.problem.plot_solution(solution, ax=ax)
                ax.set_title(f"Solution {valid_solution_count + 1}")
                valid_solution_count += 1  # Increment the valid solution counter
            except Exception as e:
                print(f"Skipping invalid solution {i + 1}: {e}")

        # Remove any unused subplots
        for j in range(valid_solution_count, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()  # Adjust layout to prevent overlap
        #plt.show()  # Show the combined figure with all valid subplots

    except Exception as e:
        print(f"Error plotting solutions: {e}")


def get_tsp_solution(bitstrings):
    # TODO Only take the first now
    bitstring = bitstrings[0]

    # Calculate n based on the length of the bitstring
    n = int(np.sqrt(len(bitstring)))
    solution = []

    for p__ in range(n):
        p_th_step = []
        for i in range(n):
            # Converting character to integer and checking if it's 1 (instead of 0.999)
            if int(bitstring[i * n + p__]) == 1:
                p_th_step.append(i)
        solution.append(p_th_step)

    return solution


def get_pos_for_graph(G:networkx.Graph):
    edges = G.edges
    # Extract all nodes
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])

    # Number of nodes
    num_nodes = len(nodes)

    # Generate positions in a circular layout
    angle_increment = 2 * np.pi / num_nodes
    pos = {}
    for i, node in enumerate(sorted(nodes)):
        angle = i * angle_increment
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos