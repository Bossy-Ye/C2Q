# Variables
PACKAGE_NAME = classical_to_quantum
PYTHON = python3
PIP = pip3
VENV_DIR = venv
TEST_DIR = classical_to_quantum/tests
EXAMPLES_DIR = examples

# Default target: install dependencies
all: install

# Create virtual environment and install dependencies
install:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_DIR)/bin/activate && $(PIP) install -r requirements.txt

# Run tests using pytest
test:
	@echo "Running tests..."
	. $(VENV_DIR)/bin/activate && pytest $(TEST_DIR)

# Run a specific example
run_example:
	@echo "Running example $(example)..."
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(EXAMPLES_DIR)/$(example).py

# Clean up the virtual environment and other generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Help
help:
	@echo "Available targets:"
	@echo "  make install       - Create virtual environment and install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make run_example example=<example_name> - Run a specific example (without .py extension)"
	@echo "  make clean         - Clean up virtual environment and other generated files"
	@echo "  make help          - Display this help message"

