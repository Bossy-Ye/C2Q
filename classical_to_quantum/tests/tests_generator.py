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
        generator = QASMGenerator()
        # Access and execute the code for the clique problem
        clique_code = data['test_cases']['clique']
        maxcut_code = data['test_cases']['maximum cut']
        eigenvalue_code = data['test_cases']['eigenvalue']
        svm_code = data['test_cases']['svm']
        cnf_code = data['test_cases']['cnf']
        self.addition_code = data['test_cases']['addition']
        independent_set_code = data['test_cases']['independent set']
        self.tsp_code = data['test_cases']['tsp']
        self.coloring_code = data['test_cases']['coloring']
        self.triangle_finding_code = data['test_cases']['triangle finding']
        self.generator = QASMGenerator()
        self.parser = ProblemParser()

    def test_triangle_finding(self):
        self.generator.qasm_generate(self.triangle_finding_code, verbose=True)
        self.assertEqual(True, True)  # add assertion here

    def test_addition(self):
        qasm = self.generator.qasm_generate(self.addition_code, verbose=True)

    def test_tsp(self):
        print(self.tsp_code)
        #self.parser.parse_code(self.tsp_code)
        #print(self.parser.evaluation())
        #qasm = self.generator.qasm_generate(self.tsp_code, verbose=True)


if __name__ == '__main__':
    unittest.main()
