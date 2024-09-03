import unittest

from qiskit import qasm2

from Framework.parser import ProblemParser
from Framework.generator import QASMGenerator
import json
import qiskit.qasm2


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

    def test_mul(self):
        qasm, _ = self.generator.qasm_generate(self.mul_code, verbose=True)
        circuit = qasm['QFT']
        print(qasm2.loads(circuit, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))

    def test_triangle_finding(self):
        print(self.triangle_finding_code)
        self.generator.qasm_generate(self.triangle_finding_code, verbose=True)
        self.assertEqual(True, True)  # add assertion here

    def test_addition(self):
        qasm = self.generator.qasm_generate(self.addition_code, verbose=True)

    def test_tsp(self):
        print(self.tsp_code)
        qasm = self.generator.qasm_generate(self.tsp_code, verbose=True)

    def test_vrp(self):
        print(self.vrp_code)
        qasm = self.generator.qasm_generate(self.vrp_code, verbose=True)

    def test_maxcut(self):
        qasm = self.generator.qasm_generate(self.maxcut_code, verbose=True)

    def test_matrix_maxcut(self):
        qasm = self.generator.qasm_generate(self.maxcut_matrix_code, verbose=True)

    def test_subtraction(self):
        print(self.sub_code)
        qasm = self.generator.qasm_generate(self.sub_code, verbose=True)

    def test_multiplication(self):
        qasm = self.generator.qasm_generate(self.mul_code, verbose=True)

    def test_cnf(self):
        qasm = self.generator.qasm_generate(self.cnf_code, verbose=True)

    def test_independent_set(self):
        print(self.independent_set_code)
        qasm = self.generator.qasm_generate(self.independent_set_code, verbose=True)

    def test_coloring(self):
        print(self.coloring_code)
        qasm = self.generator.qasm_generate(self.coloring_code, verbose=True)
        self.assertEqual(True, True)

    def test_factor(self):
        qasm = self.generator.qasm_generate(self.factor_code, verbose=True)


if __name__ == '__main__':
    unittest.main()
