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
        plt.show()  # Show the combined figure with all valid subplots

    except Exception as e:
        print(f"Error plotting solutions: {e}")