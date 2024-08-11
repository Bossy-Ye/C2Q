from classical_to_quantum.qasm_generate import QASMGenerator
classical_code = """
import numpy as np
def minimum_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.min(eigenvalues)
a= 4
b = [[1,2],[3,4],[5]]
matrix = np.array([[-2, 0, 0, -5], [0, 4, 1, 0], [0, 1, 4, 0], [-5, 0, 0, -2]])
min_eigval = minimum_eigenvalue(matrix)
print(f"The minimum eigenvalue of the matrix is: {min_eigval}")

"""

generator = QASMGenerator()

qasm = generator.qasm_generate(classical_code, verbose=True)
result = generator.run_qasm_aer(qasm, primitive='sampler', noise=0.02)
print(result)
