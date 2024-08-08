import ast

import numpy as np
from enum import Enum

ml_dic = ['SVC']


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


class ProblemParser(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.comments = []
        self.imports = []
        self.main_logic = []
        self.source = None
        self.observable = None
        self.matrix = None
        self.is_eigenvalue_problem = None
        self.in_observable_construction = None
        self.observable_code = None
        self.problem_type = None
        self.arithmetic_operator = None
        self.variables = {}
        self.assign_dispatch_table = {
            ast.Name: self.handle_assign_name,
            ast.Tuple: self.handle_assign_tuple,
            ast.List: self.handle_assign_List
        }

    def handle_assign_name(self, node):
        var_name = node.targets[0].id
        self.variables[var_name] = extract_variable(node.value)
        if var_name == 'matrix' or var_name == 'observable':
            self.problem_type = ProblemType.EIGENVALUE
            self.observable = self.variables[var_name]
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
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {'eigvals', 'eigvalsh', 'eigh'}:
                self.problem_type = ProblemType.EIGENVALUE
            if node.func.attr == 'factor':
                self.problem_type = ProblemType.FACTOR
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ml_dic:
                self.problem_type = ProblemType.MACHINELEARNING
            if 'cnf' in func_name.lower() or 'sat' in func_name.lower():
                self.problem_type = ProblemType.CNF
        self.generic_visit(node)

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

    def parse_code(self, code):
        self.source = code
        tree = ast.parse(code)
        self.visit(tree)

        return self.problem_type
