class GraphProblemError(Exception):
    """Base class for exceptions in the GraphProblem class."""
    pass


class InvalidInputError(GraphProblemError):
    """Exception raised for invalid input data."""

    def __init__(self, input_type, message="Input must be either a file path or a networkx.Graph object."):
        self.input_type = input_type
        self.message = f"{message} Received: {input_type}"
        super().__init__(self.message)


class FileLoadingError(GraphProblemError):
    """Exception raised for errors in loading a graph from a file."""

    def __init__(self, file_path, message="Error loading graph from file."):
        self.file_path = file_path
        self.message = f"{message} File: {file_path}"
        super().__init__(self.message)


class UnrecognizedProblemTypeError(GraphProblemError):
    """Exception raised for unrecognized problem types."""

    def __init__(self, problem_type, message="Unrecognized problem type."):
        self.problem_type = problem_type
        self.message = f"{message} Problem type: {problem_type}"
        super().__init__(self.message)


class NotExecutedError(Exception):
    """Exception raised when an operation is attempted before executing a required step."""

    def __init__(self,
                 message="The required operation has not been executed yet. Please execute the necessary steps first."):
        self.message = message
        super().__init__(self.message)
