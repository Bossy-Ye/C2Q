# tests/test_parser.py

import unittest
from classical_to_quantum.parser import ProblemParser
from classical_to_quantum.codegen import *
import codegen
import ast
from classical_to_quantum.codegen import *
classical_code = """
import numpy as np
def minimum_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.min(eigenvalues)
matrix = np.array([[4, 1], [2, 3]])
min_eigval = minimum_eigenvalue(matrix)
print(f"The minimum eigenvalue of the matrix is: {min_eigval}")

"""
classical_code_kernel = """
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
import json

# Load the test cases from the JSON file
with open('/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/classical_cases/c.json', 'r') as f:
    data = json.load(f)

# Access and execute the code for the clique problem
clique_code = data['test_cases']['clique']

parser = ProblemParser()
parser.parse_code(clique_code)
parser.evaluate_problem_type()
#print(parser.evaluation())
#exec(clique_code)
# #parser = ProblemParser()
# parser = ProblemParser()
# m = parser.parse_code(classical_code_kernel)
# parser.evaluate_problem_type()
# print(parser.problem_type)
# print(parser.evaluation())
# print(parser.visitor.variables)
#
# #print(parser.parse_code(classical_code))
