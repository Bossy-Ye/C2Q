from classical_to_quantum.qasm_generate import QASMGenerator

classical_code = """
def is_satisfiable(cnf_formula, assignment):
    for clause in cnf_formula:
        satisfied = False
        for literal in clause:
            var = abs(literal)
            if (literal > 0 and assignment[var]) or (literal < 0 and not assignment[var]):
                satisfied = True
                break
        if not satisfied:
            return False
    return True

def solve_cnf(cnf_formula, num_vars, assignment=None, var=1):
    if assignment is None:
        assignment = [None] * (num_vars + 1)

    if var > num_vars:
        if is_satisfiable(cnf_formula, assignment):
            return assignment
        return None

    assignment[var] = True
    result = solve_cnf(cnf_formula, num_vars, assignment, var + 1)
    if result:
        return result
        
    assignment[var] = False
    result = solve_cnf(cnf_formula, num_vars, assignment, var + 1)
    if result:
        return result

    return None

# Example usage
if __name__ == "__main__":
    cnf_formula = [
        [-1, -2, -3],
        [1, -2, 3],
        [1, 2, -3],
        [1, -2, -3],
        [-1, 2, 3],
    ]

    num_vars = 3  # Number of variables in the formula

    result = solve_cnf(cnf_formula, num_vars)
    if result:
        print("Satisfiable assignment found:")
        for i in range(1, num_vars + 1):
            print(f"x{i} = {result[i]}")
    else:
        print("No satisfiable assignment exists.")
"""

generator = QASMGenerator()

qasm = generator.qasm_generate(classical_code, verbose=True)
res = generator.run_qasm(qasm)
print(res)


