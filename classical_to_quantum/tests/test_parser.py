# tests/test_parser.py

import unittest
from classical_to_quantum.parser import ProblemParser
from classical_to_quantum.codegen import *
import codegen
import ast
from classical_to_quantum.codegen import *
import json
import utils

# Load the test cases from the JSON file
with open('/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/classical_cases/cases.json',
          'r') as f:
    data = json.load(f)

# Access and execute the code for the clique problem
clique_code = data['test_cases']['clique']
maxcut_code = data['test_cases']['maxcut']
eigenvalue_code = data['test_cases']['eigenvalue']
svm_code = data['test_cases']['svm']
cnf_code = data['test_cases']['cnf']

parser = ProblemParser()
parser.parse_code(clique_code)
parser.evaluate_problem_type()
print(parser.problem_type, parser.specific_graph_problem)


parser.parse_code(maxcut_code)
parser.evaluate_problem_type()
print(parser.problem_type, parser.specific_graph_problem)


parser.parse_code(eigenvalue_code)
parser.evaluate_problem_type()
print(parser.problem_type, parser.specific_graph_problem, parser.data)
observable = utils.decompose_into_pauli(parser.data)
print(observable)

parser.parse_code(svm_code)
parser.evaluate_problem_type()
print(parser.problem_type, parser.specific_graph_problem, parser.data)


parser.parse_code(cnf_code)
parser.evaluate_problem_type()
print(parser.problem_type, parser.specific_graph_problem, parser.data)


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
