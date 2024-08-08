import ast
import inspect
import openai


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = set()
        self.function_descriptions = {}

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        # Analyze function docstring to infer purpose
        docstring = ast.get_docstring(node)
        if docstring:
            self.function_descriptions[node.name] = docstring
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
        self.generic_visit(node)

    def analyze_code(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return self


def get_code_summary(code):
    analyzer = CodeAnalyzer().analyze_code(code)

    summary = {
        "functions": analyzer.functions,
        "classes": analyzer.classes,
        "imports": analyzer.imports,
        "variables": list(analyzer.variables),
        "function_descriptions": analyzer.function_descriptions
    }

    return summary


# Example usage:
if __name__ == "__main__":
    example_code = """
import math

def calculate_area(radius):
    \"\"\"Calculate the area of a circle.\"\"\"
    area = math.pi * radius ** 2
    return area

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def get_area(self):
        \"\"\"Get the area of the circle.\"\"\"
        return calculate_area(self.radius)
"""
    summary = get_code_summary(example_code)
    print(summary)