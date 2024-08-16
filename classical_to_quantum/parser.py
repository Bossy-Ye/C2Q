import ast

import networkx as nx
import numpy as np
from enum import Enum
from classical_to_quantum.classiq_exceptions import *

ml_dic = ['SVC']
eigenvalue_keywords = ["matrix", "observable", "eigvals", "eigvalsh", "eigh"]
ml_keywords = ["SVC", "RandomForest", "MLP", "LinearRegression", 'SVM']
cnf_keywords = ["cnf", "sat"]
graph_keywords = ["graph", "networkx", "nx", "adjacency", "nodes", "edges", "shortest_path", "dijkstra"]
clique_keywords = ["clique"]
maxcut_keywords = ["maxcut"]

def extract_matrix(node):
    matrix = []
    for row in node.elts:
        matrix_row = []
        for num in row.elts:
            if isinstance(num, ast.Constant):
                matrix_row.append(num.value)
            elif isinstance(num, ast.UnaryOp) and isinstance(num.op, ast.USub) and isinstance(num.operand,
                                                                                              ast.Constant):
                matrix_row.append(-num.operand.value)
            else:
                raise ValueError("Unexpected node type in matrix definition")
        matrix.append(matrix_row)
    return np.array(matrix)


def extract_variable(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand,
                                                                                      ast.Constant):
        return -node.operand.value
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [extract_variable(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == 'array' and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {'np', 'numpy'}:
                return extract_matrix(node.args[0])
    return 0


class ProblemType(Enum):
    EIGENVALUE = 1
    MACHINELEARNING = 2
    GRAPH = 3
    CNF = 4
    FACTOR = 5


def handle_constant(node):
    return node.value


class ProblemParser:
    def __init__(self):
        self.visitor = None
        self.problem_type = None
        self.source_code = None
        self.data = None

    def parse_code(self, code):
        self.visitor = MyVisitor()
        self.source_code = code
        self.visitor.visit(ast.parse(code))

    def evaluate_problem_type(self):
        """ Evaluate the problem type based on parsed data and also fetch data"""
        if self.visitor is None:
            raise NotParsedError
        # Check variables for problem type
        for var_name in self.visitor.variables:
            if var_name in eigenvalue_keywords:
                self._set_problem_type(ProblemType.EIGENVALUE)
            elif var_name in graph_keywords:
                self._set_problem_type(ProblemType.GRAPH)

        # Check function definitions for problem type
        for func in self.visitor.functions:
            if any(arg in eigenvalue_keywords for arg in func['args']):
                self._set_problem_type(ProblemType.EIGENVALUE)
            elif any(keyword in func['name'].lower() for keyword in ml_keywords):
                self._set_problem_type(ProblemType.MACHINELEARNING)
            elif any(keyword in func['name'].lower() for keyword in cnf_keywords):
                self._set_problem_type(ProblemType.CNF)
            elif any(keyword in func['name'].lower() for keyword in graph_keywords):
                self._set_problem_type(ProblemType.GRAPH)

                # Check main logic for problem type
                for node in self.visitor.main_logic:
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Attribute) and func.attr in eigenvalue_keywords:
                            self._set_problem_type(ProblemType.EIGENVALUE)
                        elif isinstance(func, ast.Name):
                            func_name = func.id
                            if func_name in ml_keywords:
                                self._set_problem_type(ProblemType.MACHINELEARNING)
                            elif any(keyword in func_name.lower() for keyword in cnf_keywords):
                                self._set_problem_type(ProblemType.CNF)
                            elif any(keyword in func_name.lower() for keyword in graph_keywords):
                                self._set_problem_type(ProblemType.GRAPH)
        if self.problem_type == ProblemType.GRAPH:
            self._determine_specific_graph_problem()
        # Extract corresponding data based on the detected problem type
        self.extract_data()

        # If no problem type was detected
        if not self.problem_type:
            return "No specific problem type detected."
        return f"Detected Problem Type: {self.problem_type.name}"

    def _determine_specific_graph_problem(self):
        """Determine the specific graph problem type if it's a graph problem."""
        # Check for clique-related keywords
        if any(keyword in self.visitor.variables or keyword in self.visitor.calls for keyword in clique_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "Clique Problem"
        # Check for max-cut-related keywords
        elif any(keyword in self.visitor.variables or keyword in self.visitor.calls for keyword in maxcut_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "Max-Cut Problem"
        # Add other specific graph problem checks here

    def evaluation(self):
        """ Evaluate the parsed code and provide a summary """
        if self.visitor is None:
            raise NotParsedError

        # Ensure only one problem type is detected
        problem_type_report = self.evaluate_problem_type()

        # Summary report
        report = []

        # 1. Summary of Imports
        if self.visitor.imports:
            report.append("Imports:")
            for imp in self.visitor.imports:
                report.append(f" - {imp}")
        else:
            report.append("No imports found.")

        # 2. Summary of Functions
        if self.visitor.functions:
            report.append("\nFunctions Defined:")
            for func in self.visitor.functions:
                func_info = f"Function '{func['name']}' with arguments {func['args']}"
                if func['docstring']:
                    func_info += f" - Docstring: \"{func['docstring']}\""
                report.append(f" - {func_info}")
        else:
            report.append("No functions defined.")

        # 3. Summary of Comments
        if self.visitor.comments:
            report.append("\nComments:")
            for comment in self.visitor.comments:
                report.append(f" - {comment}")
        else:
            report.append("No comments found.")

        # 4. Summary of Main Logic
        if self.visitor.main_logic:
            report.append("\nMain Logic:")
            for stmt in self.visitor.main_logic:
                report.append(f" - {ast.dump(stmt)}")
        else:
            report.append("No main logic found.")

        # 5. Summary of Variables
        if self.visitor.variables:
            report.append("\nVariables:")
            for var_name, value in self.visitor.variables.items():
                report.append(f" - {var_name}: {value}")
        else:
            report.append("No variables found.")

        # 6. Summary of Function Calls
        if self.visitor.calls:
            report.append("\nFunction Calls:")
            for call in self.visitor.calls:
                args = ", ".join(call['args'])
                report.append(f" - {call['func_name']}({args})")
        else:
            report.append("No function calls found.")

        # 7. Problem Type Detected
        report.append(f"\n{problem_type_report}")

        # Join the report into a single string
        final_report = "\n".join(report)
        return final_report

    def _set_problem_type(self, problem_type):
        """Set the problem type if not already set, or raise an error if there's a conflict"""
        if not self.problem_type:
            self.problem_type = problem_type
        elif self.problem_type != problem_type:
            raise ValueError(f"Multiple problem types detected: {self.problem_type.name} and {problem_type.name}")

    def extract_data(self):
        """Extract relevant data based on the detected problem type"""
        if self.problem_type == ProblemType.EIGENVALUE:
            # Try to find 'matrix' or 'observable'
            matrix = self.visitor.variables.get('matrix')
            observable = self.visitor.variables.get('observable')

            if matrix is not None:
                self.data = matrix
            elif observable is not None:
                self.data = observable
        elif self.problem_type == ProblemType.MACHINELEARNING:
            # TODO
            pass
        elif self.problem_type == ProblemType.CNF:
            # For CNF problems, look for the relevant CNF formula representation
            self.data = (self.visitor.variables.get('cnf') or self.visitor.variables.get(
                '3sat') or self.visitor.variables.get('cnf_formula') or
                         self.visitor.variables.get('formula'))
        elif self.problem_type == ProblemType.GRAPH:
            # Extract relevant graph data, such as nodes and edges
            graph_data = {'nodes': self.visitor.variables.get('nodes', []),
                          'edges': self.visitor.variables.get('edges', [])}
            G = nx.Graph()
            G.add_edges_from(graph_data.get("edges"))
            print(G.edges)
            print(G.nodes)
        # Add other cases for GRAPH, FACTOR, etc.
        # TODO


class MyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.comments = []
        self.imports = []
        self.main_logic = []
        self.variables = {}
        self.calls = []
        self.assign_dispatch_table = {
            ast.Name: self.handle_assign_name,
            ast.Tuple: self.handle_assign_tuple,
            ast.List: self.handle_assign_List
        }

    def handle_assign_name(self, node):
        var_name = node.targets[0].id
        self.variables[var_name] = extract_variable(node.value)
        self.generic_visit(node)

    def handle_assign_tuple(self, node):
        for elt in node.targets[0].elts:
            if isinstance(elt, ast.Name):
                var_name = elt.id
                self.variables[var_name] = extract_variable(node.value)
        self.generic_visit(node)

    def handle_assign_List(self, node):
        var_name = node.targets[0].id
        self.variables[var_name] = extract_matrix(node.values)

    def handle_unknown_assign(self, node):
        self.generic_visit(node)

    def visit_Assign(self, node):
        target_type = type(node.targets[0])
        handler = self.assign_dispatch_table.get(target_type, self.handle_unknown_assign)
        handler(node)

    def visit_Call(self, node):
        # Capture details about the function call
        call_info = {
            'func_name': self._get_function_name(node.func),
            'args': [ast.dump(arg) for arg in node.args]
        }
        self.calls.append(call_info)
        self.generic_visit(node)

    def _get_function_name(self, node):
        """Helper method to extract the function name from a call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_function_name(node.value)
            if isinstance(value, str):
                return f"{value}.{node.attr}"
            else:
                return node.attr
        return "<unknown>"

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        self.functions.append({
            'name': node.name,
            'docstring': docstring,
            'args': [arg.arg for arg in node.args.args]
        })
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Str):
            self.comments.append(node.value.s)
        self.generic_visit(node)

    def visit_Module(self, node):
        for n in node.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Str):
                self.comments.append(n.value.s)
            elif isinstance(n, ast.FunctionDef):
                self.visit_FunctionDef(n)
            elif isinstance(n, ast.Import):
                self.visit_Import(n)
            elif isinstance(n, ast.ImportFrom):
                self.visit_ImportFrom(n)
            else:
                self.main_logic.append(n)
        self.generic_visit(node)
