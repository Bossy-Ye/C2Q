import unittest
from classical_to_quantum.parser import ProblemParser
from classical_to_quantum.qasm_generate import QASMGenerator
import json
import utils


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Load the test cases from the JSON file
        with open(
                '/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/classical_cases/cases.json',
                'r') as f:
            data = json.load(f)
        # Access and execute the code for the clique problem
        self.clique_code = data['test_cases']['clique']
        self.maxcut_matrix_code = data['test_cases']['maximum cut distance matrix']
        self.maxcut_code = data['test_cases']['maximum cut']
        self.eigenvalue_code = data['test_cases']['eigenvalue']
        self.svm_code = data['test_cases']['svm']
        self.cnf_code = data['test_cases']['cnf']
        self.addition_code = data['test_cases']['addition']
        self.independent_set_code = data['test_cases']['independent set']
        self.tsp_code = data['test_cases']['tsp']
        self.coloring_code = data['test_cases']['coloring']
        self.triangle_finding_code = data['test_cases']['triangle finding']
        self.vrp_code = data['test_cases']['vrp']
        self.sub_code = data['test_cases']['subtraction']
        self.mul_code = data['test_cases']['multiplication']
        self.factor_code = data['test_cases']['factor']
        self.generator = QASMGenerator()
        self.parser = ProblemParser()

    def test_triangle_finding(self):
        self.generator.qasm_generate(self.triangle_finding_code, verbose=True)
        self.assertEqual(True, True)  # add assertion here

    def test_addition(self):
        qasm = self.generator.qasm_generate(self.addition_code, verbose=True)

    def test_tsp(self):
        qasm = self.generator.qasm_generate(self.tsp_code, verbose=True)

    def test_vrp(self):
        qasm = self.generator.qasm_generate(self.vrp_code, verbose=True)

    def test_maxcut(self):
        qasm = self.generator.qasm_generate(self.maxcut_code, verbose=True)

    def test_matrix_maxcut(self):
        qasm = self.generator.qasm_generate(self.maxcut_matrix_code, verbose=True)

    def test_subtraction(self):
        qasm = self.generator.qasm_generate(self.sub_code, verbose=True)

    def test_multiplication(self):
        qasm = self.generator.qasm_generate(self.mul_code, verbose=True)

    def test_cnf(self):
        qasm = self.generator.qasm_generate(self.cnf_code, verbose=True)

    def test_independent_set(self):
        qasm = self.generator.qasm_generate(self.independent_set_code, verbose=True)

    def test_coloring(self):
        qasm = self.generator.qasm_generate(self.coloring_code, verbose=True)
        self.assertEqual(True, True)

    def test_factor(self):
        qasm = self.generator.qasm_generate(self.factor_code, verbose=True)

if __name__ == '__main__':
    unittest.main()
