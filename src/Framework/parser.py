import ast
import re

import networkx as nx
import numpy as np
from enum import Enum
from src.classiq_exceptions import *
import operator

ml_dic = ['SVC']
eigenvalue_keywords = ["observable", "eigvals", "eigvalsh", "eigh", "eigenvalues"]
ml_keywords = ["SVC", "RandomForest", "MLP", "LinearRegression", 'SVM']
cnf_keywords = ["cnf", "sat"]
graph_keywords = ["graph", "networkx", "nx", "nodes", "edges",
                  "distance_matrix", "maximum cut", "tsp",
                  "vertex cover", "path", "distance"]
clique_keywords = ["clique"]
maxcut_keywords = ["maxcut", "max_cut", "maximum cut"]
independent_set_keywords = ["independent_set"]
tsp_keywords = ["tsp", "traveling_salesman", "tsp_brute_force", "TSP"]
vrp_keywords = ["vrp", "vehicle_routing"]
coloring_keywords = ["coloring", "colored", "color"]
arithmetic_keywords = ["addition", "subtraction", "multiplication", "sum", "minus", "add", "multiply", "mul", "sub"]
triangle_finding_keywords = ["triangle", "triangles", "find_triangle", "find_triangles", "triangle_find"]
factorization_keywords = ["factorization", "factorizations", "factorization", "factor", "gcd", "lcm", "factorize"]
verbose = False


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
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -node.operand.value
        elif isinstance(node.op, ast.UAdd):  # Handle Unary Plus (UAdd)
            return extract_variable(node.operand)
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [extract_variable(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == 'array' and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {'np', 'numpy'}:
                return extract_matrix(node.args[0])
    if isinstance(node, ast.BinOp):
        # Extract the left and right operands
        left_value = extract_variable(node.left)
        right_value = extract_variable(node.right)

        # Debugging output
        if verbose:
            print(f"Evaluating: {left_value} with {right_value}")

        # Determine the operation type and apply it
        if isinstance(node.op, ast.Add):
            return operator.add(left_value, right_value)
        elif isinstance(node.op, ast.Sub):
            return operator.sub(left_value, right_value)
        elif isinstance(node.op, ast.Mult):
            return operator.mul(left_value, right_value)
        elif isinstance(node.op, ast.Div):
            return operator.truediv(left_value, right_value)
        elif isinstance(node.op, ast.MatMult):  # Matrix multiplication (Python 3.5+)
            return operator.matmul(left_value, right_value)
        elif isinstance(node.op, ast.Mod):
            return operator.mod(left_value, right_value)
        # Add more operations as needed TODO
    if verbose:
        print(f"Unhandled node type or value: {ast.dump(node)}")

    return 0


class ProblemType(Enum):
    EIGENVALUE = 1
    MACHINELEARNING = 2
    GRAPH = 3
    CNF = 4
    FACTOR = 5
    ARITHMETICS = 6


def handle_constant(node):
    return node.value


class ProblemParser:
    def __init__(self):
        self.composite_numer = None
        self.arithmetic_arguments = None
        self.visitor = None
        self.problem_type = None
        self.source_code = None
        self.data = None
        self.specific_graph_problem = None
        self.specific_arithmetic_operation = None

    def parse_code(self, code):
        self.__init__()
        self.visitor = MyVisitor()
        self.source_code = code
        self.visitor.visit(ast.parse(code))
        self.evaluate_problem_type()

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
            elif var_name in factorization_keywords:
                self._set_problem_type(ProblemType.FACTOR)

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
            elif any(keyword in func['name'].lower() for keyword in arithmetic_keywords):
                self._set_problem_type(ProblemType.ARITHMETICS)
            elif any(keyword in func['name'].lower() for keyword in factorization_keywords):
                self._set_problem_type(ProblemType.FACTOR)

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
                    elif any(keyword in func_name.lower() for keyword in factorization_keywords):
                        self._set_problem_type(ProblemType.FACTOR)

        # Check imports for problem type
        for imp in self.visitor.imports:
            if 'networkx' in imp or 'nx' in imp:
                self._set_problem_type(ProblemType.GRAPH)
            elif 'sklearn' in imp:
                self._set_problem_type(ProblemType.MACHINELEARNING)
            elif any(eigen in imp for eigen in eigenvalue_keywords):
                self._set_problem_type(ProblemType.EIGENVALUE)
            elif 'sat' in imp or 'cnf' in imp:
                self._set_problem_type(ProblemType.CNF)
            elif any(factor in imp for factor in factorization_keywords):
                self._set_problem_type(ProblemType.FACTOR)

        def extract_variable_names_from_strings(name_strings):
            variable_names = []
            for name_string in name_strings:
                # Use regex to find the content between "id='" and "',"
                match = re.search(r"id='(.*?)'", name_string)
                if match:
                    variable_names.append(match.group(1))
            return variable_names

        # Check function calls for arithmetic operations
        for call in self.visitor.calls:
            if call['func_name'] in ['addition', 'subtraction', 'multiplication', 'division',
                                     'add', 'subtract', 'multiply', 'divide','mul',
                                     'sub', ]:
                self._set_problem_type(ProblemType.ARITHMETICS)
                arguments_string = call.get('args')
                if len(arguments_string) != 2:
                    raise ValueError("Basic arithmetic problems require only two operands")

                arguments = extract_variable_names_from_strings(arguments_string)

                self.arithmetic_arguments = arguments
                break
            elif call['func_name'].lower() in factorization_keywords:
                self._set_problem_type(ProblemType.FACTOR)
                arguments_string = call.get('args')
                if len(arguments_string) != 1:
                    raise ValueError("Factorization problems require only 1 operands")
                composite_numer = extract_variable_names_from_strings(arguments_string)

                self.composite_numer = composite_numer
                break

        if self.problem_type == ProblemType.GRAPH:
            self._determine_specific_graph_problem()
        elif self.problem_type == ProblemType.ARITHMETICS:
            self._determine_arithmetic_operation()

        # Extract corresponding data based on the detected problem type
        self.extract_data()

        # If no problem type was detected
        if not self.problem_type:
            return "No specific problem type detected."
        return f"Detected Problem Type: {self.problem_type.name}"

    def _determine_arithmetic_operation(self):
        """Determine the specific arithmetic operation type based on the variables or function calls."""

        # Keywords for each arithmetic operation
        addition_keywords = ['add', 'addition', 'plus', '+']
        subtraction_keywords = ['subtraction', 'sub', 'subtract', 'minus', '-']
        multiplication_keywords = ['multiplication', 'mul', 'multiply', 'times','*']
        division_keywords = ['division', 'divide', '/']

        # Check for addition-related keywords or operations
        if any(call['func_name'] in addition_keywords for call in self.visitor.calls):
            self.specific_arithmetic_operation = "Addition"

        # Check for subtraction-related keywords or operations
        elif any(call['func_name'] in subtraction_keywords for call in self.visitor.calls):
            self.specific_arithmetic_operation = "Subtraction"

        # Check for multiplication-related keywords or operations
        elif any(call['func_name'] in multiplication_keywords for call in self.visitor.calls):
            self.specific_arithmetic_operation = "Multiplication"

        # Check for division-related keywords or operations
        elif any(call['func_name'] in division_keywords for call in self.visitor.calls):
            self.specific_arithmetic_operation = "Division"

    def _determine_specific_graph_problem(self):
        """Determine the specific graph problem type if it's a graph problem."""

        # Function to check if any keyword is a substring in a list of strings
        def vars_contains_keyword(vars, keywords):
            return any(any(keyword in var for keyword in keywords) for var in vars)

        def calls_contains_keyword(calls, keywords):
            return any(any(keyword in call.get("func_name") for keyword in keywords) for call in calls)

        # Check for clique-related keywords
        if vars_contains_keyword(self.visitor.variables, clique_keywords) or calls_contains_keyword(self.visitor.calls,
                                                                                                    clique_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "Clique Problem"

        # Check for max-cut-related keywords
        elif vars_contains_keyword(self.visitor.variables, maxcut_keywords) or calls_contains_keyword(
                self.visitor.calls,
                maxcut_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "MaximumCut"

        # Check for independent set keywords
        elif vars_contains_keyword(self.visitor.variables, independent_set_keywords) or calls_contains_keyword(
                self.visitor.calls,
                independent_set_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "MIS"

        # Check for TSP-related keywords
        elif vars_contains_keyword(self.visitor.variables, tsp_keywords) or calls_contains_keyword(self.visitor.calls,
                                                                                                   tsp_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "TSP"

        # Check for graph coloring keywords
        elif vars_contains_keyword(self.visitor.variables, coloring_keywords) or calls_contains_keyword(
                self.visitor.calls,
                coloring_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "KColor"

        # Check for triangle finding keywords
        elif vars_contains_keyword(self.visitor.variables, triangle_finding_keywords) or calls_contains_keyword(
                self.visitor.calls,
                triangle_finding_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "Triangle"
        # Check for vrp keywords
        elif vars_contains_keyword(self.visitor.variables, vrp_keywords) or calls_contains_keyword(self.visitor.calls,
                                                                                                   vrp_keywords):
            self.problem_type = ProblemType.GRAPH
            self.specific_graph_problem = "VRP"

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
            # List of possible variable names for nodes and edges
            possible_edge_names = ['edges', 'elists', 'edge', 'Edge']
            possible_node_names = ['nodes', 'nodelist', 'node', 'Node']
            possible_adj_matrix_names = ['adjacency_matrix', 'adj_matrix', 'adjacent_matrix',
                                         'graph', 'matrix', 'distance_matrix']
            # Extract graph data
            graph_data = {}
            for edge_name in possible_edge_names:
                edges = self.visitor.variables.get(edge_name)
                if edges:
                    graph_data['edges'] = edges
                    break

            for node_name in possible_node_names:
                nodes = self.visitor.variables.get(node_name)
                if nodes:
                    graph_data['nodes'] = nodes
                    break
            for adj_matrix_name in possible_adj_matrix_names:
                adj_matrix = self.visitor.variables.get(adj_matrix_name)
                if adj_matrix is not None:
                    graph_data['adjacency_matrix'] = adj_matrix
                    break
            # If edges were found, construct the graph
            if 'edges' in graph_data:
                G = nx.Graph()
                edges_list = graph_data['edges']
                default_weight = 1
                # Convert list of lists to list of tuples, handling missing weights
                edges = [(edge[0], edge[1], edge[2]) if len(edge) == 3 else (edge[0], edge[1], default_weight) for edge
                         in edges_list]
                G.add_weighted_edges_from(edges)
                self.data = G  # Store the graph in self.data if needed
                return  # Return if edges found
            # If adjacency matrix was found, construct the graph using the matrix
            elif 'adjacency_matrix' in graph_data:
                adj_matrix = np.array(graph_data['adjacency_matrix'])
                G = nx.from_numpy_array(adj_matrix)
                self.data = G  # Store the graph in self.data if needed.
                return  # Return if matrix found
        elif self.problem_type == ProblemType.ARITHMETICS:
            data = {"left": self.visitor.variables.get(self.arithmetic_arguments[0]),
                    "right": self.visitor.variables.get(self.arithmetic_arguments[1])}
            self.data = data
        elif self.problem_type == ProblemType.FACTOR:
            data = {"composite number": self.visitor.variables.get(self.composite_numer[0])}
            self.data = data
            # Add additional checks or data extraction for other graph-related needs
            # TODO: Add other cases for GRAPH, FACTOR, etc.


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
        # Assign with a function call like matrix=np.array([...])
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
        func_name = self._get_function_name(node.func)
        call_info = {
            'func_name': func_name,
            'args': [ast.dump(arg) for arg in node.args]
        }
        self.calls.append(call_info)

        # Check if this is a call to add_edges_from and extract the edges
        if func_name.endswith('add_edges_from') and len(node.args) == 1:
            edges = extract_variable(node.args[0])
            if isinstance(edges, list):
                self.variables["edges"] = edges

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
