# Classi|Q> Project!!!
Trying to bridge the gap between classical computing and quantum computing, especially in the context of **NP-complete problem**, for those who are not familiar with quantum computing. This project is addressing both a theoretical and practical needs. Provide a general overview of the problems that can be solved by quantum computers from the perspective of **computational complexity theory**.


## License
This project takes Qiskit open source project as a reference, thanks to community.
A copy of this license is included in the `LICENSE` file.

## Attribution
- Author: Boshuai Ye, email: boshuaiye@gmail.com
- License: http://www.apache.org/licenses/LICENSE-2.0
- Qiskit: https://qiskit.org/

### Workflow of Classi|Q>

![alt text](./assets/workflow.png "Title")

### How we translate classical problem into quantum???
We aim to analyze the given classical code by extracting its Abstract Syntax Tree (AST), traversing it to identify the type of problem being solved, and then capturing the original data. 
The next step is to convert this input data into a format suitable for 
quantum computation. Currently, we are focusing on converting NP problems 
to CNF (Conjunctive Normal Form) and utilizing the Quantum Approximate 
Optimization Algorithm (QAOA). For these cases, oracles tailored to 
different types of problems will be required. And also a translator that gives readable output.

### From the point of view of computational theory
 - bounded-error quantum polynomial time (BQP) is the class of decision problems sovable by a quantum computer in polynomial time.

### Limitedness
1. ***Inherent complexity to translate original problem into CNF***...
2. ***Not all problems have a quantum counterpart...*** 
3. ***From the view of computational theory, parts of supported algorithms, say Grover, takes $\sqrt{N}$ steps, although it's a speedup compared with normal N, but in computational theory, its not a speedup at all... The square of root of a exponential function is still exponential.***
5. Difficulty in Classical Simulation:
6. Simulating the output distribution of a random quantum circuit on a classical computer is believed to be a hard problem, particularly as the number of qubits and the circuit depth increase. This hardness stems from the exponential growth of the state space (with the number of qubits) and the complex interference patterns that arise due to the random gates.

### Translating Classical Problems into Quantum

1. **Problem Formulation and Data Representation**
   - **Classical**: We aim to analyze the given classical code by extracting its Abstract Syntax Tree (AST), traversing it to identify the type of problem being solved and fetching input data
   - **Quantum**: Convert input data into quantum format (with qubits)
2. **Algorithm Selection**
   - **Classical**: Choose an appropriate classical algorithm (e.g., sorting, searching, optimization).
   - **Quantum**: Identify or design a quantum algorithm (e.g., Grover's algorithm for searching, Shor's algorithm for factoring) that can solve the problem more efficiently using quantum principles.
3. **State Initialization**
   - **Classical**: Initialize the data in a particular state or configuration.
   - **Quantum**: Prepare an initial quantum state, often starting from |0⟩, |1⟩, or a superposition state.
4. **Circuit Design**
   - **Classical**: Some parts of the algorithm are ran classically, for example: gradient decent in QAOA
   - **Quantum**: Design a quantum circuit that implements the quantum algorithm. This involves choosing quantum gates (e.g., Hadamard, CNOT, T-gate) that manipulate qubits according to the problem’s requirements. This part is automated.
5. **Execution**
   - **Local**: Run the circuit with local simulator
   - **Quantum**: Execute the quantum circuit on a remote quantum computer.
6. **Interpretation of Results**
   - **Quantum**: Interpret the measured quantum state to understand the solution. 

# Workflow of our framework
![Image of Workflow](workflow.png)
- We aim to analyze the given classical code by extracting its Abstract Syntax Tree (AST), traversing it to identify the type of problem being solved, and then capturing the original data. The next step is to convert this input data into a format suitable for quantum computation. Currently, we are focusing on converting NP problems to CNF (Conjunctive Normal Form) and utilizing the Quantum Approximate Optimization Algorithm (QAOA). For these cases, oracles tailored to different types of problems will be required. And also a translator that gives readable output.
# Outline
- **Parser**
  - Leverage Python’s Abstract Syntax Tree (AST) to perform a Depth-First Search (DFS), systematically analyzing the entire tree structure to gather all the necessary information for the Generator.
![Image of Workflow](https://i0.wp.com/pybit.es/wp-content/uploads/2021/05/tree-sketch.png?w=750&ssl=1)
  - Notice: Try to make sure your code contains only a single function, only one usage at once. And also try to make code structure clear and names of variables and functions clearly indicates its usages. This tree-based parser is not that clever yet (based mainly on rules)... I am thinking to employ OPENAI interfaces later... 
- **Generator**
   - Generate the corresponding QASM code and visualize the results, with local simulation if specified.
   - Based on results from parser and select corresponding algorithms... 
- **Optimizer**
   -  The codes generated by the translator are fed into the optimizer to determine the most suitable algorithm for running on real quantum hardware. The optimizer also takes responsibility for finding the optimal algorithm parameters, such as the number of iterations in Grover’s algorithm. If necessary, the code will be executed on local simulators.
- **Recommender** 
   - Given QASM code from Generator, selects the most suitable and available quantum computer to execute the translated quantum code and receives the quantum results.
- **Interpreter**
   - Transform the results from remote quantum computer into readable format.
- **Basic arithmetic operation**: +, -, * are implemented with quantum unitary gates
- **Graph problems**: perhaps easy to understand
  - qaoa (partially done based on openqaoa)
    - Visualization
    - Support different file format e.g., gset, tsplib 
  - grover (oracle for each problem needed)
    - Convert it to SAT problem if it could be done.
    - Why **Sat**? Sat problem was the first problem proven to be NP-complete by Stephen Cook in 1971 (Cook’s theorem). This means that every problem in NP can be           reduced to SAT, making it a kind of "universal" problem for NP. Any problem in NP can be reduced to it in polynomial time. So study SAT counterpart of a graph problem has an empirical implication.
    - How to choose an optimal iterations number wisely? We suppose the number of solutions is unknown, The formula for updating T is given by: choose T=1 initially, then $T = \lceil \frac{4}{5} T \rceil$ during each iteration[^1]
- **Eigen values (minimum or multiple)**:
- **Satisfiable Problems**:
- **Factorization**:

### Future improvements
- Utilize Pennylane to enhance visualization of ease coding[^2]
- Employ the power of randomness??? For example, in quantum oracle, the verification of CNF, only select a small number of clauses to reduce the complexity of circuits (should be really carefully handled and analysed...)
- Support more qubits, thus more complex circuits...

[^1]: Boyer, M., Brassard, G., Høyer, P., & Tapp, A. (1998). Tight bounds on quantum searching. Fortschritte Der Physik, 46(4–5), 493–505. https://doi.org/10.1002/(sici)1521-3978(199806)46:4/5

[^2]: Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed, Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer, Zeyue Niu, Antal Száva, and Nathan Killoran. PennyLane: Automatic differentiation of hybrid quantum-classical computations. 2018. arXiv:1811.04968


# Get started
### Run the following in your terminal
`make install`


# Import necessary libs


```python
import matplotlib.pyplot as plt
from qiskit import qasm2
from applications.graph.graph_problem import GraphProblem
from classical_to_quantum.applications.graph.grover_applications.graph_oracle import *
from classical_to_quantum.algorithms.grover import GroverWrapper
from classical_to_quantum.applications.graph.grover_applications.graph_color import GraphColor
from classical_to_quantum.applications.graph.Ising import Ising
from classical_to_quantum.applications.graph.grover_applications.grover_auxiliary import *
from classical_to_quantum.applications.graph.ising_auxiliary import *
from classical_to_quantum.qasm_generate import QASMGenerator
from qiskit.visualization import plot_histogram
import json
from qiskit import qasm2
from qiskit.primitives import Sampler
```

    /Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
      warnings.warn(


# Load test cases


```python
# Load the test cases from the JSON file
with open('/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/classical_cases/cases.json',
          'r') as f:
    data = json.load(f)

# Access and execute the code for the clique problem
clique_code = data['test_cases']['clique']
maxcut_code = data['test_cases']['maximum cut']
eigenvalue_code = data['test_cases']['eigenvalue']
svm_code = data['test_cases']['svm']
cnf_code = data['test_cases']['cnf']
addition_code = data['test_cases']['addition']
independent_set_code = data['test_cases']['independent set']
tsp_code = data['test_cases']['tsp']
coloring_code = data['test_cases']['coloring']
triangle_finding_code = data['test_cases']['triangle finding']
vrp_code = data['test_cases']['vrp']
factor_code = data['test_cases']['factor']
multiplication_code = data['test_cases']['multiplication']

generator = QASMGenerator()
```


# Example: Independent Set
- 1. Grover algorithm, from IS to conjunctive normal formula (CNF), then construct an oracle from this cnf, then do a search
- 2. QAOA (Quantum Approximate Optimization Algorithm)


```python
graph_problem = GraphProblem("/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G0")
independent_set_cnf = independent_set_to_sat(graph_problem.graph())
independent_set_oracle = cnf_to_quantum_oracle(independent_set_cnf)
def fun(state):
    return True
grover = GroverWrapper(oracle=independent_set_oracle,
                       iterations=1,
                       is_good_state=fun,
                       objective_qubits=list(range(graph_problem.num_nodes)))
is_res = grover.run(verbose=True)
display(plot_histogram(is_res.circuit_results[0], title="Grover's Oracle"))
```

    {   'assignment': '0000',
        'circuit_results': [   {   '0000': 0.1406249999999966,
                                   '0001': 0.1406249999999965,
                                   '0010': 0.1406249999999962,
                                   '0011': 0.0156249999999996,
                                   '0100': 0.1406249999999961,
                                   '0101': 0.0156249999999996,
                                   '0110': 0.0156249999999996,
                                   '0111': 0.0156249999999996,
                                   '1000': 0.1406249999999966,
                                   '1001': 0.1406249999999965,
                                   '1010': 0.0156249999999995,
                                   '1011': 0.0156249999999996,
                                   '1100': 0.0156249999999995,
                                   '1101': 0.0156249999999996,
                                   '1110': 0.0156249999999996,
                                   '1111': 0.0156249999999996}],
        'iterations': [1],
        'max_probability': 0.1406249999999966,
        'oracle_evaluation': True,
        'top_measurement': '0000'}



    
![png](presentation_files/presentation_6_1.png)
    


# TOP solutions are all correct!!!


```python
top_is_measurements = get_top_measurements(is_res, num=100)
plot_multiple_independent_sets(graph_problem.graph(), top_is_measurements)
```


    
![png](presentation_files/presentation_8_0.png)
    


# Example: graph 4-coloring
	1.	Coloring Problems to SAT: Coloring problems are first reduced to a Satisfiability Problem (SAT). A quantum oracle is then constructed, and Grover’s algorithm is applied.
	•	In the 4-coloring problem, each color is represented by two bits (e.g., 00 for red, 10 for green).
	2.	Quantum Approximate Optimization Algorithm (QAOA): A correlated cost Hamiltonian is constructed based on specific problem conditions, which is then optimized using QAOA.


```python
coloring_problem = GraphColor("/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G0", verbose=True)
coloring_grover_res = coloring_problem.run(verbose=True)
```

    Variable Qubits: [0, 1, 2, 3, 4, 5, 6, 7]
    Check Qubits: [8, 9, 10, 11, 12]
    Disagree List: [[[0, 1], [2, 3]], [[0, 1], [4, 5]], [[2, 3], [4, 5]], [[2, 3], [6, 7]], [[4, 5], [6, 7]]]
    Output Qubit: 13
    {   'assignment': '00011000',
        'circuit_results': [   {   '00000000': 0.0020751953125,
                                   '00000001': 0.0020751953125,
                                   '00000010': 0.0020751953125,
                                   '00000011': 0.0020751953125,
                                   '00000100': 0.0020751953125,
                                   '00000101': 0.0020751953125,
                                   '00000110': 0.0020751953125,
                                   '00000111': 0.0020751953125,
                                   '00001000': 0.0020751953125,
                                   '00001001': 0.0020751953125,
                                   '00001010': 0.0020751953125,
                                   '00001011': 0.0020751953125,
                                   '00001100': 0.0020751953125,
                                   '00001101': 0.0020751953125,
                                   '00001110': 0.0020751953125,
                                   '00001111': 0.0020751953125,
                                   '00010000': 0.0020751953125,
                                   '00010001': 0.0020751953125,
                                   '00010010': 0.0020751953125,
                                   '00010011': 0.0020751953125,
                                   '00010100': 0.0020751953125,
                                   '00010101': 0.0020751953125,
                                   '00010110': 0.0020751953125,
                                   '00010111': 0.0020751953125,
                                   '00011000': 0.0118408203125005,
                                   '00011001': 0.0020751953125,
                                   '00011010': 0.0020751953125,
                                   '00011011': 0.0118408203125005,
                                   '00011100': 0.0118408203125004,
                                   '00011101': 0.0020751953125,
                                   '00011110': 0.0118408203125004,
                                   '00011111': 0.0020751953125,
                                   '00100000': 0.0020751953125,
                                   '00100001': 0.0020751953125,
                                   '00100010': 0.0020751953125,
                                   '00100011': 0.0020751953125,
                                   '00100100': 0.0118408203125004,
                                   '00100101': 0.0020751953125,
                                   '00100110': 0.0020751953125,
                                   '00100111': 0.0118408203125005,
                                   '00101000': 0.0020751953125,
                                   '00101001': 0.0020751953125,
                                   '00101010': 0.0020751953125,
                                   '00101011': 0.0020751953125,
                                   '00101100': 0.0118408203125004,
                                   '00101101': 0.0118408203125005,
                                   '00101110': 0.0020751953125,
                                   '00101111': 0.0020751953125,
                                   '00110000': 0.0020751953125,
                                   '00110001': 0.0020751953125,
                                   '00110010': 0.0020751953125,
                                   '00110011': 0.0020751953125,
                                   '00110100': 0.0118408203125004,
                                   '00110101': 0.0020751953125,
                                   '00110110': 0.0118408203125004,
                                   '00110111': 0.0020751953125,
                                   '00111000': 0.0118408203125005,
                                   '00111001': 0.0118408203125005,
                                   '00111010': 0.0020751953125,
                                   '00111011': 0.0020751953125,
                                   '00111100': 0.0020751953125,
                                   '00111101': 0.0020751953125,
                                   '00111110': 0.0020751953125,
                                   '00111111': 0.0020751953125,
                                   '01000000': 0.0020751953125,
                                   '01000001': 0.0020751953125,
                                   '01000010': 0.0020751953125,
                                   '01000011': 0.0020751953125,
                                   '01000100': 0.0020751953125,
                                   '01000101': 0.0020751953125,
                                   '01000110': 0.0020751953125,
                                   '01000111': 0.0020751953125,
                                   '01001000': 0.0020751953125,
                                   '01001001': 0.0118408203125004,
                                   '01001010': 0.0020751953125,
                                   '01001011': 0.0118408203125004,
                                   '01001100': 0.0020751953125,
                                   '01001101': 0.0118408203125005,
                                   '01001110': 0.0118408203125004,
                                   '01001111': 0.0020751953125,
                                   '01010000': 0.0020751953125,
                                   '01010001': 0.0020751953125,
                                   '01010010': 0.0020751953125,
                                   '01010011': 0.0020751953125,
                                   '01010100': 0.0020751953125,
                                   '01010101': 0.0020751953125,
                                   '01010110': 0.0020751953125,
                                   '01010111': 0.0020751953125,
                                   '01011000': 0.0020751953125,
                                   '01011001': 0.0020751953125,
                                   '01011010': 0.0020751953125,
                                   '01011011': 0.0020751953125,
                                   '01011100': 0.0020751953125,
                                   '01011101': 0.0020751953125,
                                   '01011110': 0.0020751953125,
                                   '01011111': 0.0020751953125,
                                   '01100000': 0.0020751953125,
                                   '01100001': 0.0118408203125004,
                                   '01100010': 0.0020751953125,
                                   '01100011': 0.0118408203125004,
                                   '01100100': 0.0020751953125,
                                   '01100101': 0.0020751953125,
                                   '01100110': 0.0020751953125,
                                   '01100111': 0.0020751953125,
                                   '01101000': 0.0020751953125,
                                   '01101001': 0.0020751953125,
                                   '01101010': 0.0020751953125,
                                   '01101011': 0.0020751953125,
                                   '01101100': 0.0118408203125005,
                                   '01101101': 0.0118408203125005,
                                   '01101110': 0.0020751953125,
                                   '01101111': 0.0020751953125,
                                   '01110000': 0.0020751953125,
                                   '01110001': 0.0118408203125004,
                                   '01110010': 0.0118408203125005,
                                   '01110011': 0.0020751953125,
                                   '01110100': 0.0020751953125,
                                   '01110101': 0.0020751953125,
                                   '01110110': 0.0020751953125,
                                   '01110111': 0.0020751953125,
                                   '01111000': 0.0118408203125005,
                                   '01111001': 0.0118408203125005,
                                   '01111010': 0.0020751953125,
                                   '01111011': 0.0020751953125,
                                   '01111100': 0.0020751953125,
                                   '01111101': 0.0020751953125,
                                   '01111110': 0.0020751953125,
                                   '01111111': 0.0020751953125,
                                   '10000000': 0.0020751953125,
                                   '10000001': 0.0020751953125,
                                   '10000010': 0.0020751953125,
                                   '10000011': 0.0020751953125,
                                   '10000100': 0.0020751953125,
                                   '10000101': 0.0020751953125,
                                   '10000110': 0.0118408203125004,
                                   '10000111': 0.0118408203125005,
                                   '10001000': 0.0020751953125,
                                   '10001001': 0.0020751953125,
                                   '10001010': 0.0020751953125,
                                   '10001011': 0.0020751953125,
                                   '10001100': 0.0020751953125,
                                   '10001101': 0.0118408203125005,
                                   '10001110': 0.0118408203125004,
                                   '10001111': 0.0020751953125,
                                   '10010000': 0.0020751953125,
                                   '10010001': 0.0020751953125,
                                   '10010010': 0.0118408203125005,
                                   '10010011': 0.0118408203125004,
                                   '10010100': 0.0020751953125,
                                   '10010101': 0.0020751953125,
                                   '10010110': 0.0020751953125,
                                   '10010111': 0.0020751953125,
                                   '10011000': 0.0020751953125,
                                   '10011001': 0.0020751953125,
                                   '10011010': 0.0020751953125,
                                   '10011011': 0.0020751953125,
                                   '10011100': 0.0118408203125004,
                                   '10011101': 0.0020751953125,
                                   '10011110': 0.0118408203125004,
                                   '10011111': 0.0020751953125,
                                   '10100000': 0.0020751953125,
                                   '10100001': 0.0020751953125,
                                   '10100010': 0.0020751953125,
                                   '10100011': 0.0020751953125,
                                   '10100100': 0.0020751953125,
                                   '10100101': 0.0020751953125,
                                   '10100110': 0.0020751953125,
                                   '10100111': 0.0020751953125,
                                   '10101000': 0.0020751953125,
                                   '10101001': 0.0020751953125,
                                   '10101010': 0.0020751953125,
                                   '10101011': 0.0020751953125,
                                   '10101100': 0.0020751953125,
                                   '10101101': 0.0020751953125,
                                   '10101110': 0.0020751953125,
                                   '10101111': 0.0020751953125,
                                   '10110000': 0.0020751953125,
                                   '10110001': 0.0118408203125004,
                                   '10110010': 0.0118408203125005,
                                   '10110011': 0.0020751953125,
                                   '10110100': 0.0118408203125004,
                                   '10110101': 0.0020751953125,
                                   '10110110': 0.0118408203125004,
                                   '10110111': 0.0020751953125,
                                   '10111000': 0.0020751953125,
                                   '10111001': 0.0020751953125,
                                   '10111010': 0.0020751953125,
                                   '10111011': 0.0020751953125,
                                   '10111100': 0.0020751953125,
                                   '10111101': 0.0020751953125,
                                   '10111110': 0.0020751953125,
                                   '10111111': 0.0020751953125,
                                   '11000000': 0.0020751953125,
                                   '11000001': 0.0020751953125,
                                   '11000010': 0.0020751953125,
                                   '11000011': 0.0020751953125,
                                   '11000100': 0.0020751953125,
                                   '11000101': 0.0020751953125,
                                   '11000110': 0.0118408203125004,
                                   '11000111': 0.0118408203125005,
                                   '11001000': 0.0020751953125,
                                   '11001001': 0.0118408203125004,
                                   '11001010': 0.0020751953125,
                                   '11001011': 0.0118408203125004,
                                   '11001100': 0.0020751953125,
                                   '11001101': 0.0020751953125,
                                   '11001110': 0.0020751953125,
                                   '11001111': 0.0020751953125,
                                   '11010000': 0.0020751953125,
                                   '11010001': 0.0020751953125,
                                   '11010010': 0.0118408203125005,
                                   '11010011': 0.0118408203125005,
                                   '11010100': 0.0020751953125,
                                   '11010101': 0.0020751953125,
                                   '11010110': 0.0020751953125,
                                   '11010111': 0.0020751953125,
                                   '11011000': 0.0118408203125005,
                                   '11011001': 0.0020751953125,
                                   '11011010': 0.0020751953125,
                                   '11011011': 0.0118408203125004,
                                   '11011100': 0.0020751953125,
                                   '11011101': 0.0020751953125,
                                   '11011110': 0.0020751953125,
                                   '11011111': 0.0020751953125,
                                   '11100000': 0.0020751953125,
                                   '11100001': 0.0118408203125004,
                                   '11100010': 0.0020751953125,
                                   '11100011': 0.0118408203125004,
                                   '11100100': 0.0118408203125004,
                                   '11100101': 0.0020751953125,
                                   '11100110': 0.0020751953125,
                                   '11100111': 0.0118408203125005,
                                   '11101000': 0.0020751953125,
                                   '11101001': 0.0020751953125,
                                   '11101010': 0.0020751953125,
                                   '11101011': 0.0020751953125,
                                   '11101100': 0.0020751953125,
                                   '11101101': 0.0020751953125,
                                   '11101110': 0.0020751953125,
                                   '11101111': 0.0020751953125,
                                   '11110000': 0.0020751953125,
                                   '11110001': 0.0020751953125,
                                   '11110010': 0.0020751953125,
                                   '11110011': 0.0020751953125,
                                   '11110100': 0.0020751953125,
                                   '11110101': 0.0020751953125,
                                   '11110110': 0.0020751953125,
                                   '11110111': 0.0020751953125,
                                   '11111000': 0.0020751953125,
                                   '11111001': 0.0020751953125,
                                   '11111010': 0.0020751953125,
                                   '11111011': 0.0020751953125,
                                   '11111100': 0.0020751953125,
                                   '11111101': 0.0020751953125,
                                   '11111110': 0.0020751953125,
                                   '11111111': 0.0020751953125}],
        'iterations': [1],
        'max_probability': 0.0118408203125005,
        'oracle_evaluation': True,
        'top_measurement': '00011000'}


### Retrieve the top measurements that are most likely to represent solutions.


```python
bitstring_results = get_top_measurements(coloring_grover_res)
bitstring_results    
```




    [{'00011000': 0.0118408203125005},
     {'00011011': 0.0118408203125005},
     {'00100111': 0.0118408203125005}]



### Interpret the results from the quantum circuit for graph-related problems.


```python
plot_multiple_graph_colorings(coloring_problem.graph(), bitstring_results, num_per_row=3)
```

    Plotting for bitstring 00011000 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'red'}
    Plotting for bitstring 00011011 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
    Plotting for bitstring 00100111 with color assignment: {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}



    
![png](./presentation_files/presentation_14_1.png)
    


### QAOA counterpart for 4_coloring problems 


```python
coloring_problem_ising = Ising(
            "/Users/mac/workspace/quantum-journey/QUANTUM-CLASSICAL-TRANSLATION/classical_to_quantum/cases/Gset/G3",
            "KColor")
coloring_problem_ising.plot_graph()
result = coloring_problem_ising.run(verbose=False)
```

    -- cannot find parameters matching version: , using: 22.1.1.0



    
![png](presentation_files/presentation_16_1.png)
    



```python
solutions = result.most_probable_states.get('solutions_bitstrings')
plot_first_valid_coloring_solutions(solutions, coloring_problem_ising)

```

    Skipping invalid solution 1: 'c' argument has 9 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 2: 'c' argument has 7 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 4: 'c' argument has 7 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 5: 'c' argument has 7 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 7: 'c' argument has 6 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 8: 'c' argument has 7 elements, which is inconsistent with 'x' and 'y' with size 5.



    
![png](presentation_files/presentation_17_1.png)
    


# Classical bruteforce solutions for independent set and its recommended quantum solutions: quantum oracle and QAOA are given


```python
qasm_code = generator.qasm_generate(classical_code=independent_set_code, verbose=True)
grover_code = qasm_code.get('grover')
qaoa_code = qasm_code.get('qaoa')
```

    problem type: ProblemType.GRAPH data: Graph with 5 nodes and 5 edges
    -------graph problem type:MIS--------
    <class 'classical_to_quantum.applications.graph.Ising.Ising'>
    -- cannot find parameters matching version: , using: 22.1.1.0
    {'angles': [0.584652534436, 0.35210653055, 0.11899483725, 0.119186527067, 0.349748055969, 0.585024721328], 'cost': -1.87, 'measurement_outcomes': {'01100': 1, '10101': 1, '00001': 2, '10011': 2, '10010': 9, '01001': 4, '00110': 14, '00100': 3, '10000': 6, '10100': 12, '00000': 1, '01000': 7, '00101': 2, '10110': 18, '00010': 2, '10001': 16}, 'job_id': '85e41dc8-f4a3-4541-b552-b5f1ecf314cf', 'eval_number': 132}



    
![png](presentation_files/presentation_19_1.png)
    


    <class 'classical_to_quantum.applications.graph.graph_problem.GraphProblem'>
    {   'assignment': '00010',
        'circuit_results': [   {   '00000': 0.0703124999999972,
                                   '00001': 0.0703124999999972,
                                   '00010': 0.0703124999999974,
                                   '00011': 0.0078124999999997,
                                   '00100': 0.0703124999999974,
                                   '00101': 0.0703124999999974,
                                   '00110': 0.0078124999999997,
                                   '00111': 0.0078124999999997,
                                   '01000': 0.0703124999999972,
                                   '01001': 0.0703124999999972,
                                   '01010': 0.0078124999999998,
                                   '01011': 0.0078124999999997,
                                   '01100': 0.0703124999999974,
                                   '01101': 0.0703124999999974,
                                   '01110': 0.0078124999999997,
                                   '01111': 0.0078124999999997,
                                   '10000': 0.0703124999999973,
                                   '10001': 0.0703124999999973,
                                   '10010': 0.0703124999999974,
                                   '10011': 0.0078124999999997,
                                   '10100': 0.0078124999999998,
                                   '10101': 0.0078124999999998,
                                   '10110': 0.0078124999999997,
                                   '10111': 0.0078124999999997,
                                   '11000': 0.0078124999999998,
                                   '11001': 0.0078124999999998,
                                   '11010': 0.0078124999999998,
                                   '11011': 0.0078124999999997,
                                   '11100': 0.0078124999999998,
                                   '11101': 0.0078124999999998,
                                   '11110': 0.0078124999999997,
                                   '11111': 0.0078124999999997}],
        'iterations': [1],
        'max_probability': 0.0703124999999974,
        'oracle_evaluation': True,
        'top_measurement': '00010'}


## Generated QASM code


```python
grover_circuit = qasm2.loads(grover_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
qaoa_circuit = qasm2.loads(qaoa_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
print(qaoa_circuit)
print(grover_circuit)
```

            ┌───┐               ┌───────┐  ┌─────────────┐              »
     q11_0: ┤ H ├─■─────────────┤ Rz(0) ├──┤ Rx(-1.1925) ├──────────────»
            ├───┤ │ZZ(0.1115)   └───────┘  └─────────────┘┌────────────┐»
     q11_1: ┤ H ├─■────────────■─────────────■────────────┤ Rz(-0.223) ├»
            ├───┤              │ZZ(0.1115)   │            └────────────┘»
     q11_2: ┤ H ├──────────────■─────────────┼─────────────■────────────»
            ├───┤                            │ZZ(0.1115)   │            »
     q11_3: ┤ H ├────────────────────────────■─────────────┼────────────»
            ├───┤                                          │ZZ(0.1115)  »
     q11_4: ┤ H ├──────────────────────────────────────────■────────────»
            └───┘                                                       »
    meas: 5/════════════════════════════════════════════════════════════»
                                                                        »
    «                                         ┌───────┐   ┌──────────────┐»
    « q11_0: ─────────────────■───────────────┤ Rz(0) ├───┤ Rx(-0.71588) ├»
    «        ┌─────────────┐  │ZZ(0.3577)     └───────┘   └──────────────┘»
    « q11_1: ┤ Rx(-1.1925) ├──■──────────────■──────────────■─────────────»
    «        ├─────────────┤┌─────────────┐  │ZZ(0.3577)    │             »
    « q11_2: ┤ Rz(-0.1115) ├┤ Rx(-1.1925) ├──■──────────────┼─────────────»
    «        └─────────────┘├─────────────┤┌─────────────┐  │ZZ(0.3577)   »
    « q11_3: ──■────────────┤ Rz(-0.1115) ├┤ Rx(-1.1925) ├──■─────────────»
    «          │ZZ(0.1115)  ├─────────────┤├─────────────┤                »
    « q11_4: ──■────────────┤ Rz(-0.1115) ├┤ Rx(-1.1925) ├────────────────»
    «                       └─────────────┘└─────────────┘                »
    «meas: 5/═════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                                           ┌───────┐    »
    « q11_0: ──────────────────────────────────■────────────────┤ Rz(0) ├────»
    «        ┌──────────────┐┌──────────────┐  │ZZ(0.59595)     └───────┘    »
    « q11_1: ┤ Rz(-0.71541) ├┤ Rx(-0.71588) ├──■───────────────■─────────────»
    «        └──────────────┘├─────────────┬┘┌──────────────┐  │ZZ(0.59595)  »
    « q11_2: ──■─────────────┤ Rz(-0.3577) ├─┤ Rx(-0.71588) ├──■─────────────»
    «          │             └─────────────┘ ├─────────────┬┘┌──────────────┐»
    « q11_3: ──┼───────────────■─────────────┤ Rz(-0.3577) ├─┤ Rx(-0.71588) ├»
    «          │ZZ(0.3577)     │ZZ(0.3577)   ├─────────────┤ ├──────────────┤»
    « q11_4: ──■───────────────■─────────────┤ Rz(-0.3577) ├─┤ Rx(-0.71588) ├»
    «                                        └─────────────┘ └──────────────┘»
    «meas: 5/════════════════════════════════════════════════════════════════»
    «                                                                        »
    «        ┌─────────────┐                                               »
    « q11_0: ┤ Rx(-0.2385) ├───────────────────────────────────────────────»
    «        └─────────────┘┌─────────────┐┌─────────────┐                 »
    « q11_1: ──■────────────┤ Rz(-1.1919) ├┤ Rx(-0.2385) ├─────────────────»
    «          │            └─────────────┘├─────────────┴┐┌─────────────┐ »
    « q11_2: ──┼──────────────■────────────┤ Rz(-0.59595) ├┤ Rx(-0.2385) ├─»
    «          │ZZ(0.59595)   │            └──────────────┘├─────────────┴┐»
    « q11_3: ──■──────────────┼──────────────■─────────────┤ Rz(-0.59595) ├»
    «                         │ZZ(0.59595)   │ZZ(0.59595)  ├──────────────┤»
    « q11_4: ─────────────────■──────────────■─────────────┤ Rz(-0.59595) ├»
    «                                                      └──────────────┘»
    «meas: 5/══════════════════════════════════════════════════════════════»
    «                                                                      »
    «                        ░ ┌─┐            
    « q11_0: ────────────────░─┤M├────────────
    «                        ░ └╥┘┌─┐         
    « q11_1: ────────────────░──╫─┤M├─────────
    «                        ░  ║ └╥┘┌─┐      
    « q11_2: ────────────────░──╫──╫─┤M├──────
    «        ┌─────────────┐ ░  ║  ║ └╥┘┌─┐   
    « q11_3: ┤ Rx(-0.2385) ├─░──╫──╫──╫─┤M├───
    «        ├─────────────┤ ░  ║  ║  ║ └╥┘┌─┐
    « q11_4: ┤ Rx(-0.2385) ├─░──╫──╫──╫──╫─┤M├
    «        └─────────────┘ ░  ║  ║  ║  ║ └╥┘
    «meas: 5/═══════════════════╩══╩══╩══╩══╩═
    «                           0  1  2  3  4 
          ┌───┐┌──────────────────────┐┌──────────────────────┐┌─┐            
     q_0: ┤ H ├┤0                     ├┤0                     ├┤M├────────────
          ├───┤│                      ││                      │└╥┘┌─┐         
     q_1: ┤ H ├┤1                     ├┤1                     ├─╫─┤M├─────────
          ├───┤│                      ││                      │ ║ └╥┘┌─┐      
     q_2: ┤ H ├┤2                     ├┤2                     ├─╫──╫─┤M├──────
          ├───┤│                      ││                      │ ║  ║ └╥┘┌─┐   
     q_3: ┤ H ├┤3                     ├┤3                     ├─╫──╫──╫─┤M├───
          ├───┤│                      ││                      │ ║  ║  ║ └╥┘┌─┐
     q_4: ┤ H ├┤4                     ├┤4                     ├─╫──╫──╫──╫─┤M├
          └───┘│                      ││                      │ ║  ║  ║  ║ └╥┘
     q_5: ─────┤5  Gate_q_11337495408 ├┤5  Gate_q_11388920112 ├─╫──╫──╫──╫──╫─
               │                      ││                      │ ║  ║  ║  ║  ║ 
     q_6: ─────┤6                     ├┤6                     ├─╫──╫──╫──╫──╫─
               │                      ││                      │ ║  ║  ║  ║  ║ 
     q_7: ─────┤7                     ├┤7                     ├─╫──╫──╫──╫──╫─
               │                      ││                      │ ║  ║  ║  ║  ║ 
     q_8: ─────┤8                     ├┤8                     ├─╫──╫──╫──╫──╫─
               │                      ││                      │ ║  ║  ║  ║  ║ 
     q_9: ─────┤9                     ├┤9                     ├─╫──╫──╫──╫──╫─
               │                      ││                      │ ║  ║  ║  ║  ║ 
    q_10: ─────┤10                    ├┤10                    ├─╫──╫──╫──╫──╫─
               └──────────────────────┘└──────────────────────┘ ║  ║  ║  ║  ║ 
     c: 5/══════════════════════════════════════════════════════╩══╩══╩══╩══╩═
                                                                0  1  2  3  4 



```python
grover_circuit = qasm2.loads(grover_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
qaoa_circuit = qasm2.loads(qaoa_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
print(qaoa_circuit)
print(grover_circuit)
```

            ┌───┐                ┌───────┐   ┌─────────────┐                »
     q14_0: ┤ H ├─■──────────────┤ Rz(0) ├───┤ Rx(-1.1693) ├────────────────»
            ├───┤ │ZZ(0.11919)   └───────┘   └─────────────┘┌──────────────┐»
     q14_1: ┤ H ├─■─────────────■──────────────■────────────┤ Rz(-0.23837) ├»
            ├───┤               │ZZ(0.11919)   │            └──────────────┘»
     q14_2: ┤ H ├───────────────■──────────────┼──────────────■─────────────»
            ├───┤                              │ZZ(0.11919)   │             »
     q14_3: ┤ H ├──────────────────────────────■──────────────┼─────────────»
            ├───┤                                             │ZZ(0.11919)  »
     q14_4: ┤ H ├─────────────────────────────────────────────■─────────────»
            └───┘                                                           »
    meas: 5/════════════════════════════════════════════════════════════════»
                                                                            »
    «                                           ┌───────┐   ┌──────────────┐»
    « q14_0: ──────────────────■────────────────┤ Rz(0) ├───┤ Rx(-0.70421) ├»
    «        ┌─────────────┐   │ZZ(0.34975)     └───────┘   └──────────────┘»
    « q14_1: ┤ Rx(-1.1693) ├───■───────────────■──────────────■─────────────»
    «        ├─────────────┴┐┌─────────────┐   │ZZ(0.34975)   │             »
    « q14_2: ┤ Rz(-0.11919) ├┤ Rx(-1.1693) ├───■──────────────┼─────────────»
    «        └──────────────┘├─────────────┴┐┌─────────────┐  │ZZ(0.34975)  »
    « q14_3: ──■─────────────┤ Rz(-0.11919) ├┤ Rx(-1.1693) ├──■─────────────»
    «          │ZZ(0.11919)  ├──────────────┤├─────────────┤                »
    « q14_4: ──■─────────────┤ Rz(-0.11919) ├┤ Rx(-1.1693) ├────────────────»
    «                        └──────────────┘└─────────────┘                »
    «meas: 5/═══════════════════════════════════════════════════════════════»
    «                                                                       »
    «                                                          ┌───────┐    »
    « q14_0: ─────────────────────────────────■────────────────┤ Rz(0) ├────»
    «        ┌─────────────┐┌──────────────┐  │ZZ(0.58502)     └───────┘    »
    « q14_1: ┤ Rz(-0.6995) ├┤ Rx(-0.70421) ├──■───────────────■─────────────»
    «        └─────────────┘├──────────────┤┌──────────────┐  │ZZ(0.58502)  »
    « q14_2: ──■────────────┤ Rz(-0.34975) ├┤ Rx(-0.70421) ├──■─────────────»
    «          │            └──────────────┘├──────────────┤┌──────────────┐»
    « q14_3: ──┼──────────────■─────────────┤ Rz(-0.34975) ├┤ Rx(-0.70421) ├»
    «          │ZZ(0.34975)   │ZZ(0.34975)  ├──────────────┤├──────────────┤»
    « q14_4: ──■──────────────■─────────────┤ Rz(-0.34975) ├┤ Rx(-0.70421) ├»
    «                                       └──────────────┘└──────────────┘»
    «meas: 5/═══════════════════════════════════════════════════════════════»
    «                                                                       »
    «        ┌──────────────┐                                              »
    « q14_0: ┤ Rx(-0.23799) ├──────────────────────────────────────────────»
    «        └──────────────┘┌───────────┐ ┌──────────────┐                »
    « q14_1: ──■─────────────┤ Rz(-1.17) ├─┤ Rx(-0.23799) ├────────────────»
    «          │             └───────────┘ ├──────────────┤┌──────────────┐»
    « q14_2: ──┼──────────────■────────────┤ Rz(-0.58502) ├┤ Rx(-0.23799) ├»
    «          │ZZ(0.58502)   │            └──────────────┘├──────────────┤»
    « q14_3: ──■──────────────┼──────────────■─────────────┤ Rz(-0.58502) ├»
    «                         │ZZ(0.58502)   │ZZ(0.58502)  ├──────────────┤»
    « q14_4: ─────────────────■──────────────■─────────────┤ Rz(-0.58502) ├»
    «                                                      └──────────────┘»
    «meas: 5/══════════════════════════════════════════════════════════════»
    «                                                                      »
    «                         ░ ┌─┐            
    « q14_0: ─────────────────░─┤M├────────────
    «                         ░ └╥┘┌─┐         
    « q14_1: ─────────────────░──╫─┤M├─────────
    «                         ░  ║ └╥┘┌─┐      
    « q14_2: ─────────────────░──╫──╫─┤M├──────
    «        ┌──────────────┐ ░  ║  ║ └╥┘┌─┐   
    « q14_3: ┤ Rx(-0.23799) ├─░──╫──╫──╫─┤M├───
    «        ├──────────────┤ ░  ║  ║  ║ └╥┘┌─┐
    « q14_4: ┤ Rx(-0.23799) ├─░──╫──╫──╫──╫─┤M├
    «        └──────────────┘ ░  ║  ║  ║  ║ └╥┘
    «meas: 5/════════════════════╩══╩══╩══╩══╩═
    «                            0  1  2  3  4 
          ┌───┐┌─────────────────────┐┌──────────────────────┐┌─┐            
     q_0: ┤ H ├┤0                    ├┤0                     ├┤M├────────────
          ├───┤│                     ││                      │└╥┘┌─┐         
     q_1: ┤ H ├┤1                    ├┤1                     ├─╫─┤M├─────────
          ├───┤│                     ││                      │ ║ └╥┘┌─┐      
     q_2: ┤ H ├┤2                    ├┤2                     ├─╫──╫─┤M├──────
          ├───┤│                     ││                      │ ║  ║ └╥┘┌─┐   
     q_3: ┤ H ├┤3                    ├┤3                     ├─╫──╫──╫─┤M├───
          ├───┤│                     ││                      │ ║  ║  ║ └╥┘┌─┐
     q_4: ┤ H ├┤4                    ├┤4                     ├─╫──╫──╫──╫─┤M├
          └───┘│                     ││                      │ ║  ║  ║  ║ └╥┘
     q_5: ─────┤5  Gate_q_6288549728 ├┤5  Gate_q_11344763440 ├─╫──╫──╫──╫──╫─
               │                     ││                      │ ║  ║  ║  ║  ║ 
     q_6: ─────┤6                    ├┤6                     ├─╫──╫──╫──╫──╫─
               │                     ││                      │ ║  ║  ║  ║  ║ 
     q_7: ─────┤7                    ├┤7                     ├─╫──╫──╫──╫──╫─
               │                     ││                      │ ║  ║  ║  ║  ║ 
     q_8: ─────┤8                    ├┤8                     ├─╫──╫──╫──╫──╫─
               │                     ││                      │ ║  ║  ║  ║  ║ 
     q_9: ─────┤9                    ├┤9                     ├─╫──╫──╫──╫──╫─
               │                     ││                      │ ║  ║  ║  ║  ║ 
    q_10: ─────┤10                   ├┤10                    ├─╫──╫──╫──╫──╫─
               └─────────────────────┘└──────────────────────┘ ║  ║  ║  ║  ║ 
     c: 5/═════════════════════════════════════════════════════╩══╩══╩══╩══╩═
                                                               0  1  2  3  4 


# An example for coloring problem, where generator recommends grover algorithm and QAOA


```python
qasm_code = generator.qasm_generate(classical_code=coloring_code, verbose=True)
grover_code = qasm_code.get('grover')
qaoa_code = qasm_code.get('qaoa')
grover_circuit = qasm2.loads(grover_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
qaoa_circuit = qasm2.loads(qaoa_code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
print(qaoa_circuit)
print(grover_circuit)
```

    problem type: ProblemType.GRAPH data: Graph with 5 nodes and 5 edges
    -------graph problem type:KColor--------
    <class 'classical_to_quantum.applications.graph.Ising.Ising'>
    {'angles': [0.61675153255, 0.359419972508, 0.12251366554, 0.111505963289, 0.348747016668, 0.568025873603], 'cost': 180.92, 'measurement_outcomes': {'10000100000110000010': 1, '00000010100000001000': 1, '00101000000110000000': 1, '00010011000110010100': 1, '11000100110010100100': 1, '00100110011000001000': 1, '01000000000110000100': 1, '00100000010000000001': 1, '10001010100001000000': 1, '00100100000000001000': 1, '01000100010110000100': 1, '00101000000001001110': 1, '00110000010100010001': 1, '00000101001010000010': 1, '00111000000001000010': 1, '11110010101100000010': 1, '00001011101010010110': 1, '00100000100100000001': 1, '00010100100000010010': 1, '10000001000001000100': 1, '00100000100001000001': 1, '00000111110000100010': 1, '00010011010100010001': 1, '01100001000001011000': 1, '01001000000000111000': 1, '00010110100000100001': 1, '10000100000100100000': 1, '00100010001000010100': 1, '00010001100101000000': 1, '00101000000101001000': 1, '01000001100000001110': 1, '01000000101010000000': 1, '01100010001000100010': 1, '01000000111000100001': 1, '00100110101001010010': 1, '01001000000100100010': 1, '01010001010001010000': 1, '01010100110001000010': 1, '00100001010010000001': 1, '00010000001001000010': 1, '00011010010010000101': 1, '00100000010010000010': 1, '00100101100000101000': 1, '00111000101100000100': 1, '00010000011001010100': 1, '00010010100001000010': 1, '00011000100010000010': 1, '00100010001100000100': 1, '00100000010010001000': 1, '00001000011100010011': 1, '00010100001010000010': 1, '00100110001010000010': 1, '00010100001010000001': 1, '00010001000100010100': 1, '00011010110000001001': 1, '10000010000001000001': 1, '10001000100001000001': 1, '00010001011010000010': 1, '00101001010000101000': 1, '00000000111100000001': 1, '10101000101010001001': 1, '10000000000110000001': 1, '00000100000100100001': 1, '10000010001000100001': 1, '10001000010100110100': 1, '00001100000000010010': 1, '01000101010101101010': 1, '01000000100100000000': 1, '01001000010101000010': 1, '00000001010000100010': 1, '00011000001100010001': 1, '10000001100000100001': 1, '10101000100010000000': 1, '10001000100000010100': 1, '00101001000010000001': 1, '00110010011000000001': 1, '00100000100010000010': 1, '00101000000100100000': 1, '00100011000010110010': 1, '10000110000000011000': 1, '01000001000001000010': 1, '10000000010110110010': 1, '00001000001001000000': 1, '00010010000010000010': 1, '01000000000100010001': 1, '00100100000110000100': 1, '01100000010000001000': 1, '00010010000001000001': 1, '10000000010001010000': 1, '00100100100011001000': 1, '01011000010001000010': 1, '01000001001000001000': 1, '10010000000000010100': 1, '01000010000100010100': 1, '01010100010010000001': 1, '01010000000000010010': 1, '01000101011100000001': 1, '10000100000100000010': 1, '01000110010001000100': 1, '00010010100100000010': 1}, 'job_id': '5b555dd3-98c4-47df-b75c-ac086e59b310', 'eval_number': 260}
    ['10000100000110000010', '00000010100000001000', '00101000000110000000', '00010011000110010100', '11000100110010100100', '00100110011000001000', '01000000000110000100', '00100000010000000001', '10001010100001000000', '00100100000000001000', '01000100010110000100', '00101000000001001110', '00110000010100010001', '00000101001010000010', '00111000000001000010', '11110010101100000010', '00001011101010010110', '00100000100100000001', '00010100100000010010', '10000001000001000100', '00100000100001000001', '00000111110000100010', '00010011010100010001', '01100001000001011000', '01001000000000111000', '00010110100000100001', '10000100000100100000', '00100010001000010100', '00010001100101000000', '00101000000101001000', '01000001100000001110', '01000000101010000000', '01100010001000100010', '01000000111000100001', '00100110101001010010', '01001000000100100010', '01010001010001010000', '01010100110001000010', '00100001010010000001', '00010000001001000010', '00011010010010000101', '00100000010010000010', '00100101100000101000', '00111000101100000100', '00010000011001010100', '00010010100001000010', '00011000100010000010', '00100010001100000100', '00100000010010001000', '00001000011100010011', '00010100001010000010', '00100110001010000010', '00010100001010000001', '00010001000100010100', '00011010110000001001', '10000010000001000001', '10001000100001000001', '00010001011010000010', '00101001010000101000', '00000000111100000001', '10101000101010001001', '10000000000110000001', '00000100000100100001', '10000010001000100001', '10001000010100110100', '00001100000000010010', '01000101010101101010', '01000000100100000000', '01001000010101000010', '00000001010000100010', '00011000001100010001', '10000001100000100001', '10101000100010000000', '10001000100000010100', '00101001000010000001', '00110010011000000001', '00100000100010000010', '00101000000100100000', '00100011000010110010', '10000110000000011000', '01000001000001000010', '10000000010110110010', '00001000001001000000', '00010010000010000010', '01000000000100010001', '00100100000110000100', '01100000010000001000', '00010010000001000001', '10000000010001010000', '00100100100011001000', '01011000010001000010', '01000001001000001000', '10010000000000010100', '01000010000100010100', '01010100010010000001', '01010000000000010010', '01000101011100000001', '10000100000100000010', '01000110010001000100', '00010010100100000010']
    Skipping invalid solution 2: 'c' argument has 3 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 3: 'c' argument has 4 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 4: 'c' argument has 7 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 5: 'c' argument has 8 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 6: 'c' argument has 6 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 7: 'c' argument has 4 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 8: 'c' argument has 3 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 10: 'c' argument has 3 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 11: 'c' argument has 6 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 12: 'c' argument has 6 elements, which is inconsistent with 'x' and 'y' with size 5.
    Skipping invalid solution 13: 'c' argument has 6 elements, which is inconsistent with 'x' and 'y' with size 5.



    
![png](presentation_files/presentation_24_1.png)
    


    <class 'classical_to_quantum.applications.graph.grover_applications.graph_color.GraphColor'>
    {   'assignment': '0001000110',
        'circuit_results': [   {   '0000000000': 0.0016925632953658,
                                   '0000000001': 0.0005100071430206,
                                   '0000000010': 0.0005100071430206,
                                   '0000000011': 0.0005100071430206,
                                   '0000000100': 0.0016925632953657,
                                   '0000000101': 0.0016925632953656,
                                   '0000000110': 0.0005100071430207,
                                   '0000000111': 0.0005100071430207,
                                   '0000001000': 0.0016925632953657,
                                   '0000001001': 0.0005100071430206,
                                   '0000001010': 0.0016925632953656,
                                   '0000001011': 0.0005100071430206,
                                   '0000001100': 0.0016925632953657,
                                   '0000001101': 0.0005100071430206,
                                   '0000001110': 0.0005100071430207,
                                   '0000001111': 0.0016925632953656,
                                   '0000010000': 0.0016925632953656,
                                   '0000010001': 0.0016925632953657,
                                   '0000010010': 0.0005100071430206,
                                   '0000010011': 0.0005100071430206,
                                   '0000010100': 0.0005100071430207,
                                   '0000010101': 0.0016925632953658,
                                   '0000010110': 0.0005100071430207,
                                   '0000010111': 0.0005100071430207,
                                   '0000011000': 0.0005100071430206,
                                   '0000011001': 0.0016925632953657,
                                   '0000011010': 0.0016925632953656,
                                   '0000011011': 0.0005100071430206,
                                   '0000011100': 0.0005100071430206,
                                   '0000011101': 0.0016925632953657,
                                   '0000011110': 0.0005100071430206,
                                   '0000011111': 0.0016925632953656,
                                   '0000100000': 0.0016925632953657,
                                   '0000100001': 0.0005100071430206,
                                   '0000100010': 0.0016925632953656,
                                   '0000100011': 0.0005100071430206,
                                   '0000100100': 0.0005100071430206,
                                   '0000100101': 0.0016925632953657,
                                   '0000100110': 0.0016925632953657,
                                   '0000100111': 0.0005100071430206,
                                   '0000101000': 0.0005100071430207,
                                   '0000101001': 0.0005100071430207,
                                   '0000101010': 0.0016925632953658,
                                   '0000101011': 0.0005100071430207,
                                   '0000101100': 0.0005100071430206,
                                   '0000101101': 0.0005100071430206,
                                   '0000101110': 0.0016925632953657,
                                   '0000101111': 0.0016925632953657,
                                   '0000110000': 0.0016925632953657,
                                   '0000110001': 0.0005100071430206,
                                   '0000110010': 0.0005100071430206,
                                   '0000110011': 0.0016925632953657,
                                   '0000110100': 0.0005100071430206,
                                   '0000110101': 0.0016925632953657,
                                   '0000110110': 0.0005100071430206,
                                   '0000110111': 0.0016925632953657,
                                   '0000111000': 0.0005100071430206,
                                   '0000111001': 0.0005100071430206,
                                   '0000111010': 0.0016925632953657,
                                   '0000111011': 0.0016925632953657,
                                   '0000111100': 0.0005100071430207,
                                   '0000111101': 0.0005100071430207,
                                   '0000111110': 0.0005100071430207,
                                   '0000111111': 0.0016925632953658,
                                   '0001000000': 0.0005100071430206,
                                   '0001000001': 0.0005100071430207,
                                   '0001000010': 0.0005100071430207,
                                   '0001000011': 0.0005100071430207,
                                   '0001000100': 0.0005100071430206,
                                   '0001000101': 0.001692563295366,
                                   '0001000110': 0.0028751194477114,
                                   '0001000111': 0.0028751194477114,
                                   '0001001000': 0.0005100071430206,
                                   '0001001001': 0.0028751194477114,
                                   '0001001010': 0.001692563295366,
                                   '0001001011': 0.0028751194477114,
                                   '0001001100': 0.0005100071430206,
                                   '0001001101': 0.0028751194477114,
                                   '0001001110': 0.0028751194477114,
                                   '0001001111': 0.001692563295366,
                                   '0001010000': 0.001692563295366,
                                   '0001010001': 0.0005100071430206,
                                   '0001010010': 0.0005100071430207,
                                   '0001010011': 0.0005100071430207,
                                   '0001010100': 0.0005100071430207,
                                   '0001010101': 0.0005100071430207,
                                   '0001010110': 0.0005100071430207,
                                   '0001010111': 0.0005100071430207,
                                   '0001011000': 0.0005100071430207,
                                   '0001011001': 0.0005100071430206,
                                   '0001011010': 0.001692563295366,
                                   '0001011011': 0.0005100071430207,
                                   '0001011100': 0.0005100071430207,
                                   '0001011101': 0.0005100071430206,
                                   '0001011110': 0.0005100071430207,
                                   '0001011111': 0.001692563295366,
                                   '0001100000': 0.001692563295366,
                                   '0001100001': 0.0028751194477114,
                                   '0001100010': 0.0005100071430206,
                                   '0001100011': 0.0028751194477114,
                                   '0001100100': 0.0028751194477114,
                                   '0001100101': 0.001692563295366,
                                   '0001100110': 0.0005100071430206,
                                   '0001100111': 0.0028751194477114,
                                   '0001101000': 0.0005100071430207,
                                   '0001101001': 0.0005100071430207,
                                   '0001101010': 0.0005100071430206,
                                   '0001101011': 0.0005100071430207,
                                   '0001101100': 0.0028751194477114,
                                   '0001101101': 0.0028751194477114,
                                   '0001101110': 0.0005100071430206,
                                   '0001101111': 0.001692563295366,
                                   '0001110000': 0.001692563295366,
                                   '0001110001': 0.0028751194477114,
                                   '0001110010': 0.0028751194477114,
                                   '0001110011': 0.0005100071430206,
                                   '0001110100': 0.0028751194477114,
                                   '0001110101': 0.001692563295366,
                                   '0001110110': 0.0028751194477114,
                                   '0001110111': 0.0005100071430206,
                                   '0001111000': 0.0028751194477114,
                                   '0001111001': 0.0028751194477114,
                                   '0001111010': 0.001692563295366,
                                   '0001111011': 0.0005100071430206,
                                   '0001111100': 0.0005100071430207,
                                   '0001111101': 0.0005100071430207,
                                   '0001111110': 0.0005100071430207,
                                   '0001111111': 0.0005100071430206,
                                   '0010000000': 0.0005100071430206,
                                   '0010000001': 0.0005100071430207,
                                   '0010000010': 0.0005100071430207,
                                   '0010000011': 0.0005100071430207,
                                   '0010000100': 0.001692563295366,
                                   '0010000101': 0.0005100071430207,
                                   '0010000110': 0.0028751194477113,
                                   '0010000111': 0.0028751194477113,
                                   '0010001000': 0.001692563295366,
                                   '0010001001': 0.0028751194477113,
                                   '0010001010': 0.0005100071430207,
                                   '0010001011': 0.0028751194477112,
                                   '0010001100': 0.001692563295366,
                                   '0010001101': 0.0028751194477113,
                                   '0010001110': 0.0028751194477113,
                                   '0010001111': 0.0005100071430207,
                                   '0010010000': 0.0005100071430207,
                                   '0010010001': 0.001692563295366,
                                   '0010010010': 0.0028751194477113,
                                   '0010010011': 0.0028751194477113,
                                   '0010010100': 0.0005100071430207,
                                   '0010010101': 0.0005100071430206,
                                   '0010010110': 0.0005100071430207,
                                   '0010010111': 0.0005100071430207,
                                   '0010011000': 0.0028751194477113,
                                   '0010011001': 0.001692563295366,
                                   '0010011010': 0.0005100071430207,
                                   '0010011011': 0.0028751194477113,
                                   '0010011100': 0.0028751194477113,
                                   '0010011101': 0.001692563295366,
                                   '0010011110': 0.0028751194477113,
                                   '0010011111': 0.0005100071430207,
                                   '0010100000': 0.0005100071430207,
                                   '0010100001': 0.0005100071430206,
                                   '0010100010': 0.0016925632953661,
                                   '0010100011': 0.0005100071430206,
                                   '0010100100': 0.0005100071430207,
                                   '0010100101': 0.0005100071430207,
                                   '0010100110': 0.0016925632953661,
                                   '0010100111': 0.0005100071430206,
                                   '0010101000': 0.0005100071430207,
                                   '0010101001': 0.0005100071430207,
                                   '0010101010': 0.0005100071430206,
                                   '0010101011': 0.0005100071430207,
                                   '0010101100': 0.0005100071430207,
                                   '0010101101': 0.0005100071430206,
                                   '0010101110': 0.0016925632953661,
                                   '0010101111': 0.0005100071430207,
                                   '0010110000': 0.0005100071430207,
                                   '0010110001': 0.0028751194477113,
                                   '0010110010': 0.0028751194477113,
                                   '0010110011': 0.001692563295366,
                                   '0010110100': 0.0028751194477113,
                                   '0010110101': 0.0005100071430207,
                                   '0010110110': 0.0028751194477113,
                                   '0010110111': 0.001692563295366,
                                   '0010111000': 0.0028751194477113,
                                   '0010111001': 0.0028751194477113,
                                   '0010111010': 0.0005100071430207,
                                   '0010111011': 0.001692563295366,
                                   '0010111100': 0.0005100071430207,
                                   '0010111101': 0.0005100071430207,
                                   '0010111110': 0.0005100071430207,
                                   '0010111111': 0.0005100071430206,
                                   '0011000000': 0.0016925632953663,
                                   '0011000001': 0.0005100071430207,
                                   '0011000010': 0.0005100071430207,
                                   '0011000011': 0.0005100071430207,
                                   '0011000100': 0.0005100071430206,
                                   '0011000101': 0.0005100071430207,
                                   '0011000110': 0.0028751194477113,
                                   '0011000111': 0.0028751194477113,
                                   '0011001000': 0.0005100071430206,
                                   '0011001001': 0.0028751194477112,
                                   '0011001010': 0.0005100071430207,
                                   '0011001011': 0.0028751194477112,
                                   '0011001100': 0.0005100071430206,
                                   '0011001101': 0.0028751194477113,
                                   '0011001110': 0.0028751194477113,
                                   '0011001111': 0.0005100071430207,
                                   '0011010000': 0.0005100071430207,
                                   '0011010001': 0.0005100071430206,
                                   '0011010010': 0.0028751194477113,
                                   '0011010011': 0.0028751194477113,
                                   '0011010100': 0.0005100071430207,
                                   '0011010101': 0.0016925632953663,
                                   '0011010110': 0.0005100071430207,
                                   '0011010111': 0.0005100071430207,
                                   '0011011000': 0.0028751194477113,
                                   '0011011001': 0.0005100071430206,
                                   '0011011010': 0.0005100071430207,
                                   '0011011011': 0.0028751194477113,
                                   '0011011100': 0.0028751194477112,
                                   '0011011101': 0.0005100071430206,
                                   '0011011110': 0.0028751194477112,
                                   '0011011111': 0.0005100071430207,
                                   '0011100000': 0.0005100071430207,
                                   '0011100001': 0.0028751194477113,
                                   '0011100010': 0.0005100071430206,
                                   '0011100011': 0.0028751194477113,
                                   '0011100100': 0.0028751194477113,
                                   '0011100101': 0.0005100071430207,
                                   '0011100110': 0.0005100071430206,
                                   '0011100111': 0.0028751194477113,
                                   '0011101000': 0.0005100071430207,
                                   '0011101001': 0.0005100071430207,
                                   '0011101010': 0.0016925632953663,
                                   '0011101011': 0.0005100071430207,
                                   '0011101100': 0.0028751194477113,
                                   '0011101101': 0.0028751194477113,
                                   '0011101110': 0.0005100071430206,
                                   '0011101111': 0.0005100071430207,
                                   '0011110000': 0.0005100071430207,
                                   '0011110001': 0.0005100071430207,
                                   '0011110010': 0.0005100071430207,
                                   '0011110011': 0.0005100071430206,
                                   '0011110100': 0.0005100071430207,
                                   '0011110101': 0.0005100071430207,
                                   '0011110110': 0.0005100071430207,
                                   '0011110111': 0.0005100071430206,
                                   '0011111000': 0.0005100071430207,
                                   '0011111001': 0.0005100071430207,
                                   '0011111010': 0.0005100071430207,
                                   '0011111011': 0.0005100071430206,
                                   '0011111100': 0.0005100071430207,
                                   '0011111101': 0.0005100071430207,
                                   '0011111110': 0.0005100071430207,
                                   '0011111111': 0.0016925632953665,
                                   '0100000000': 0.0005100071430207,
                                   '0100000001': 0.0005100071430207,
                                   '0100000010': 0.0005100071430207,
                                   '0100000011': 0.0005100071430207,
                                   '0100000100': 0.0005100071430207,
                                   '0100000101': 0.0016925632953661,
                                   '0100000110': 0.0005100071430207,
                                   '0100000111': 0.0005100071430207,
                                   '0100001000': 0.0005100071430207,
                                   '0100001001': 0.0005100071430206,
                                   '0100001010': 0.0016925632953661,
                                   '0100001011': 0.0005100071430207,
                                   '0100001100': 0.0005100071430207,
                                   '0100001101': 0.0005100071430207,
                                   '0100001110': 0.0005100071430207,
                                   '0100001111': 0.0016925632953661,
                                   '0100010000': 0.001692563295366,
                                   '0100010001': 0.0005100071430206,
                                   '0100010010': 0.0005100071430207,
                                   '0100010011': 0.0005100071430207,
                                   '0100010100': 0.0005100071430206,
                                   '0100010101': 0.0005100071430206,
                                   '0100010110': 0.0005100071430206,
                                   '0100010111': 0.0005100071430206,
                                   '0100011000': 0.0005100071430207,
                                   '0100011001': 0.0005100071430206,
                                   '0100011010': 0.0016925632953659,
                                   '0100011011': 0.0005100071430207,
                                   '0100011100': 0.0005100071430207,
                                   '0100011101': 0.0005100071430206,
                                   '0100011110': 0.0005100071430207,
                                   '0100011111': 0.0016925632953659,
                                   '0100100000': 0.001692563295366,
                                   '0100100001': 0.0005100071430207,
                                   '0100100010': 0.0005100071430206,
                                   '0100100011': 0.0005100071430207,
                                   '0100100100': 0.0005100071430207,
                                   '0100100101': 0.001692563295366,
                                   '0100100110': 0.0005100071430206,
                                   '0100100111': 0.0005100071430207,
                                   '0100101000': 0.0005100071430206,
                                   '0100101001': 0.0005100071430206,
                                   '0100101010': 0.0005100071430206,
                                   '0100101011': 0.0005100071430206,
                                   '0100101100': 0.0005100071430207,
                                   '0100101101': 0.0005100071430207,
                                   '0100101110': 0.0005100071430206,
                                   '0100101111': 0.001692563295366,
                                   '0100110000': 0.001692563295366,
                                   '0100110001': 0.0005100071430207,
                                   '0100110010': 0.0005100071430207,
                                   '0100110011': 0.0005100071430206,
                                   '0100110100': 0.0005100071430207,
                                   '0100110101': 0.001692563295366,
                                   '0100110110': 0.0005100071430207,
                                   '0100110111': 0.0005100071430206,
                                   '0100111000': 0.0005100071430207,
                                   '0100111001': 0.0005100071430207,
                                   '0100111010': 0.001692563295366,
                                   '0100111011': 0.0005100071430206,
                                   '0100111100': 0.0005100071430206,
                                   '0100111101': 0.0005100071430206,
                                   '0100111110': 0.0005100071430206,
                                   '0100111111': 0.0005100071430206,
                                   '0101000000': 0.001692563295366,
                                   '0101000001': 0.0005100071430207,
                                   '0101000010': 0.0005100071430207,
                                   '0101000011': 0.0005100071430207,
                                   '0101000100': 0.0016925632953657,
                                   '0101000101': 0.0016925632953668,
                                   '0101000110': 0.0005100071430207,
                                   '0101000111': 0.0005100071430207,
                                   '0101001000': 0.0016925632953656,
                                   '0101001001': 0.0005100071430206,
                                   '0101001010': 0.0016925632953668,
                                   '0101001011': 0.0005100071430206,
                                   '0101001100': 0.0016925632953657,
                                   '0101001101': 0.0005100071430207,
                                   '0101001110': 0.0005100071430207,
                                   '0101001111': 0.0016925632953668,
                                   '0101010000': 0.0016925632953663,
                                   '0101010001': 0.0016925632953657,
                                   '0101010010': 0.0005100071430206,
                                   '0101010011': 0.0005100071430206,
                                   '0101010100': 0.0005100071430207,
                                   '0101010101': 0.0016925632953658,
                                   '0101010110': 0.0005100071430207,
                                   '0101010111': 0.0005100071430207,
                                   '0101011000': 0.0005100071430206,
                                   '0101011001': 0.0016925632953657,
                                   '0101011010': 0.0016925632953663,
                                   '0101011011': 0.0005100071430206,
                                   '0101011100': 0.0005100071430206,
                                   '0101011101': 0.0016925632953657,
                                   '0101011110': 0.0005100071430206,
                                   '0101011111': 0.0016925632953663,
                                   '0101100000': 0.0016925632953668,
                                   '0101100001': 0.0005100071430207,
                                   '0101100010': 0.0016925632953656,
                                   '0101100011': 0.0005100071430207,
                                   '0101100100': 0.0005100071430207,
                                   '0101100101': 0.0016925632953668,
                                   '0101100110': 0.0016925632953657,
                                   '0101100111': 0.0005100071430207,
                                   '0101101000': 0.0005100071430207,
                                   '0101101001': 0.0005100071430207,
                                   '0101101010': 0.001692563295366,
                                   '0101101011': 0.0005100071430207,
                                   '0101101100': 0.0005100071430207,
                                   '0101101101': 0.0005100071430207,
                                   '0101101110': 0.0016925632953657,
                                   '0101101111': 0.0016925632953668,
                                   '0101110000': 0.0016925632953668,
                                   '0101110001': 0.0005100071430207,
                                   '0101110010': 0.0005100071430206,
                                   '0101110011': 0.0016925632953657,
                                   '0101110100': 0.0005100071430206,
                                   '0101110101': 0.0016925632953668,
                                   '0101110110': 0.0005100071430206,
                                   '0101110111': 0.0016925632953656,
                                   '0101111000': 0.0005100071430207,
                                   '0101111001': 0.0005100071430207,
                                   '0101111010': 0.0016925632953668,
                                   '0101111011': 0.0016925632953657,
                                   '0101111100': 0.0005100071430207,
                                   '0101111101': 0.0005100071430207,
                                   '0101111110': 0.0005100071430207,
                                   '0101111111': 0.001692563295366,
                                   '0110000000': 0.0016925632953664,
                                   '0110000001': 0.0005100071430207,
                                   '0110000010': 0.0005100071430207,
                                   '0110000011': 0.0005100071430207,
                                   '0110000100': 0.0005100071430206,
                                   '0110000101': 0.0028751194477113,
                                   '0110000110': 0.0005100071430207,
                                   '0110000111': 0.0005100071430207,
                                   '0110001000': 0.0005100071430206,
                                   '0110001001': 0.0005100071430207,
                                   '0110001010': 0.0028751194477113,
                                   '0110001011': 0.0005100071430207,
                                   '0110001100': 0.0005100071430206,
                                   '0110001101': 0.0005100071430207,
                                   '0110001110': 0.0005100071430207,
                                   '0110001111': 0.0028751194477113,
                                   '0110010000': 0.0028751194477113,
                                   '0110010001': 0.0005100071430206,
                                   '0110010010': 0.0005100071430207,
                                   '0110010011': 0.0005100071430207,
                                   '0110010100': 0.0005100071430207,
                                   '0110010101': 0.0016925632953664,
                                   '0110010110': 0.0005100071430207,
                                   '0110010111': 0.0005100071430207,
                                   '0110011000': 0.0005100071430207,
                                   '0110011001': 0.0005100071430206,
                                   '0110011010': 0.0028751194477113,
                                   '0110011011': 0.0005100071430207,
                                   '0110011100': 0.0005100071430207,
                                   '0110011101': 0.0005100071430206,
                                   '0110011110': 0.0005100071430207,
                                   '0110011111': 0.0028751194477113,
                                   '0110100000': 0.0005100071430207,
                                   '0110100001': 0.0005100071430206,
                                   '0110100010': 0.0005100071430207,
                                   '0110100011': 0.0005100071430206,
                                   '0110100100': 0.0005100071430206,
                                   '0110100101': 0.0005100071430207,
                                   '0110100110': 0.0005100071430207,
                                   '0110100111': 0.0005100071430206,
                                   '0110101000': 0.0005100071430207,
                                   '0110101001': 0.0005100071430207,
                                   '0110101010': 0.0016925632953665,
                                   '0110101011': 0.0005100071430207,
                                   '0110101100': 0.0005100071430206,
                                   '0110101101': 0.0005100071430206,
                                   '0110101110': 0.0005100071430207,
                                   '0110101111': 0.0005100071430207,
                                   '0110110000': 0.0028751194477113,
                                   '0110110001': 0.0005100071430207,
                                   '0110110010': 0.0005100071430207,
                                   '0110110011': 0.0005100071430206,
                                   '0110110100': 0.0005100071430207,
                                   '0110110101': 0.0028751194477113,
                                   '0110110110': 0.0005100071430207,
                                   '0110110111': 0.0005100071430206,
                                   '0110111000': 0.0005100071430207,
                                   '0110111001': 0.0005100071430207,
                                   '0110111010': 0.0028751194477113,
                                   '0110111011': 0.0005100071430206,
                                   '0110111100': 0.0005100071430207,
                                   '0110111101': 0.0005100071430207,
                                   '0110111110': 0.0005100071430207,
                                   '0110111111': 0.0016925632953664,
                                   '0111000000': 0.0005100071430206,
                                   '0111000001': 0.0005100071430207,
                                   '0111000010': 0.0005100071430207,
                                   '0111000011': 0.0005100071430207,
                                   '0111000100': 0.0016925632953659,
                                   '0111000101': 0.0028751194477113,
                                   '0111000110': 0.0005100071430207,
                                   '0111000111': 0.0005100071430207,
                                   '0111001000': 0.0016925632953659,
                                   '0111001001': 0.0005100071430207,
                                   '0111001010': 0.0028751194477113,
                                   '0111001011': 0.0005100071430207,
                                   '0111001100': 0.0016925632953659,
                                   '0111001101': 0.0005100071430207,
                                   '0111001110': 0.0005100071430207,
                                   '0111001111': 0.0028751194477113,
                                   '0111010000': 0.0028751194477113,
                                   '0111010001': 0.0016925632953659,
                                   '0111010010': 0.0005100071430207,
                                   '0111010011': 0.0005100071430207,
                                   '0111010100': 0.0005100071430207,
                                   '0111010101': 0.0005100071430206,
                                   '0111010110': 0.0005100071430207,
                                   '0111010111': 0.0005100071430207,
                                   '0111011000': 0.0005100071430207,
                                   '0111011001': 0.0016925632953659,
                                   '0111011010': 0.0028751194477113,
                                   '0111011011': 0.0005100071430207,
                                   '0111011100': 0.0005100071430207,
                                   '0111011101': 0.0016925632953659,
                                   '0111011110': 0.0005100071430207,
                                   '0111011111': 0.0028751194477113,
                                   '0111100000': 0.0028751194477113,
                                   '0111100001': 0.0005100071430207,
                                   '0111100010': 0.0016925632953659,
                                   '0111100011': 0.0005100071430207,
                                   '0111100100': 0.0005100071430207,
                                   '0111100101': 0.0028751194477113,
                                   '0111100110': 0.0016925632953659,
                                   '0111100111': 0.0005100071430207,
                                   '0111101000': 0.0005100071430207,
                                   '0111101001': 0.0005100071430207,
                                   '0111101010': 0.0005100071430206,
                                   '0111101011': 0.0005100071430207,
                                   '0111101100': 0.0005100071430207,
                                   '0111101101': 0.0005100071430207,
                                   '0111101110': 0.0016925632953659,
                                   '0111101111': 0.0028751194477113,
                                   '0111110000': 0.0005100071430207,
                                   '0111110001': 0.0005100071430207,
                                   '0111110010': 0.0005100071430207,
                                   '0111110011': 0.0016925632953661,
                                   '0111110100': 0.0005100071430207,
                                   '0111110101': 0.0005100071430207,
                                   '0111110110': 0.0005100071430207,
                                   '0111110111': 0.0016925632953661,
                                   '0111111000': 0.0005100071430207,
                                   '0111111001': 0.0005100071430207,
                                   '0111111010': 0.0005100071430207,
                                   '0111111011': 0.0016925632953661,
                                   '0111111100': 0.0005100071430207,
                                   '0111111101': 0.0005100071430207,
                                   '0111111110': 0.0005100071430207,
                                   '0111111111': 0.0005100071430207,
                                   '1000000000': 0.0005100071430206,
                                   '1000000001': 0.0005100071430206,
                                   '1000000010': 0.0005100071430206,
                                   '1000000011': 0.0005100071430206,
                                   '1000000100': 0.0016925632953659,
                                   '1000000101': 0.0005100071430206,
                                   '1000000110': 0.0005100071430207,
                                   '1000000111': 0.0005100071430207,
                                   '1000001000': 0.0016925632953659,
                                   '1000001001': 0.0005100071430207,
                                   '1000001010': 0.0005100071430206,
                                   '1000001011': 0.0005100071430207,
                                   '1000001100': 0.0016925632953659,
                                   '1000001101': 0.0005100071430207,
                                   '1000001110': 0.0005100071430207,
                                   '1000001111': 0.0005100071430206,
                                   '1000010000': 0.0005100071430206,
                                   '1000010001': 0.001692563295366,
                                   '1000010010': 0.0005100071430207,
                                   '1000010011': 0.0005100071430207,
                                   '1000010100': 0.0005100071430207,
                                   '1000010101': 0.0005100071430206,
                                   '1000010110': 0.0005100071430207,
                                   '1000010111': 0.0005100071430207,
                                   '1000011000': 0.0005100071430207,
                                   '1000011001': 0.001692563295366,
                                   '1000011010': 0.0005100071430206,
                                   '1000011011': 0.0005100071430207,
                                   '1000011100': 0.0005100071430207,
                                   '1000011101': 0.001692563295366,
                                   '1000011110': 0.0005100071430207,
                                   '1000011111': 0.0005100071430206,
                                   '1000100000': 0.0005100071430206,
                                   '1000100001': 0.0005100071430207,
                                   '1000100010': 0.0016925632953659,
                                   '1000100011': 0.0005100071430207,
                                   '1000100100': 0.0005100071430207,
                                   '1000100101': 0.0005100071430206,
                                   '1000100110': 0.001692563295366,
                                   '1000100111': 0.0005100071430207,
                                   '1000101000': 0.0005100071430207,
                                   '1000101001': 0.0005100071430207,
                                   '1000101010': 0.0005100071430206,
                                   '1000101011': 0.0005100071430207,
                                   '1000101100': 0.0005100071430207,
                                   '1000101101': 0.0005100071430207,
                                   '1000101110': 0.001692563295366,
                                   '1000101111': 0.0005100071430206,
                                   '1000110000': 0.0005100071430206,
                                   '1000110001': 0.0005100071430207,
                                   '1000110010': 0.0005100071430207,
                                   '1000110011': 0.001692563295366,
                                   '1000110100': 0.0005100071430207,
                                   '1000110101': 0.0005100071430206,
                                   '1000110110': 0.0005100071430207,
                                   '1000110111': 0.001692563295366,
                                   '1000111000': 0.0005100071430207,
                                   '1000111001': 0.0005100071430207,
                                   '1000111010': 0.0005100071430206,
                                   '1000111011': 0.001692563295366,
                                   '1000111100': 0.0005100071430207,
                                   '1000111101': 0.0005100071430207,
                                   '1000111110': 0.0005100071430207,
                                   '1000111111': 0.0005100071430206,
                                   '1001000000': 0.0016925632953664,
                                   '1001000001': 0.0005100071430207,
                                   '1001000010': 0.0005100071430207,
                                   '1001000011': 0.0005100071430207,
                                   '1001000100': 0.0028751194477114,
                                   '1001000101': 0.0005100071430206,
                                   '1001000110': 0.0005100071430206,
                                   '1001000111': 0.0005100071430206,
                                   '1001001000': 0.0028751194477114,
                                   '1001001001': 0.0005100071430206,
                                   '1001001010': 0.0005100071430206,
                                   '1001001011': 0.0005100071430206,
                                   '1001001100': 0.0028751194477114,
                                   '1001001101': 0.0005100071430206,
                                   '1001001110': 0.0005100071430206,
                                   '1001001111': 0.0005100071430206,
                                   '1001010000': 0.0005100071430207,
                                   '1001010001': 0.0005100071430207,
                                   '1001010010': 0.0005100071430207,
                                   '1001010011': 0.0005100071430207,
                                   '1001010100': 0.0005100071430207,
                                   '1001010101': 0.0016925632953665,
                                   '1001010110': 0.0005100071430207,
                                   '1001010111': 0.0005100071430207,
                                   '1001011000': 0.0005100071430207,
                                   '1001011001': 0.0005100071430207,
                                   '1001011010': 0.0005100071430207,
                                   '1001011011': 0.0005100071430207,
                                   '1001011100': 0.0005100071430207,
                                   '1001011101': 0.0005100071430207,
                                   '1001011110': 0.0005100071430207,
                                   '1001011111': 0.0005100071430207,
                                   '1001100000': 0.0005100071430206,
                                   '1001100001': 0.0005100071430206,
                                   '1001100010': 0.0028751194477114,
                                   '1001100011': 0.0005100071430206,
                                   '1001100100': 0.0005100071430207,
                                   '1001100101': 0.0005100071430206,
                                   '1001100110': 0.0028751194477114,
                                   '1001100111': 0.0005100071430206,
                                   '1001101000': 0.0005100071430207,
                                   '1001101001': 0.0005100071430207,
                                   '1001101010': 0.0016925632953664,
                                   '1001101011': 0.0005100071430207,
                                   '1001101100': 0.0005100071430207,
                                   '1001101101': 0.0005100071430206,
                                   '1001101110': 0.0028751194477114,
                                   '1001101111': 0.0005100071430206,
                                   '1001110000': 0.0005100071430206,
                                   '1001110001': 0.0005100071430206,
                                   '1001110010': 0.0005100071430206,
                                   '1001110011': 0.0028751194477114,
                                   '1001110100': 0.0005100071430206,
                                   '1001110101': 0.0005100071430206,
                                   '1001110110': 0.0005100071430206,
                                   '1001110111': 0.0028751194477114,
                                   '1001111000': 0.0005100071430206,
                                   '1001111001': 0.0005100071430206,
                                   '1001111010': 0.0005100071430206,
                                   '1001111011': 0.0028751194477114,
                                   '1001111100': 0.0005100071430207,
                                   '1001111101': 0.0005100071430207,
                                   '1001111110': 0.0005100071430207,
                                   '1001111111': 0.0016925632953664,
                                   '1010000000': 0.001692563295366,
                                   '1010000001': 0.0005100071430207,
                                   '1010000010': 0.0005100071430207,
                                   '1010000011': 0.0005100071430207,
                                   '1010000100': 0.0016925632953668,
                                   '1010000101': 0.0016925632953657,
                                   '1010000110': 0.0005100071430206,
                                   '1010000111': 0.0005100071430206,
                                   '1010001000': 0.0016925632953668,
                                   '1010001001': 0.0005100071430206,
                                   '1010001010': 0.0016925632953657,
                                   '1010001011': 0.0005100071430206,
                                   '1010001100': 0.0016925632953668,
                                   '1010001101': 0.0005100071430206,
                                   '1010001110': 0.0005100071430206,
                                   '1010001111': 0.0016925632953657,
                                   '1010010000': 0.0016925632953657,
                                   '1010010001': 0.0016925632953668,
                                   '1010010010': 0.0005100071430206,
                                   '1010010011': 0.0005100071430206,
                                   '1010010100': 0.0005100071430207,
                                   '1010010101': 0.001692563295366,
                                   '1010010110': 0.0005100071430207,
                                   '1010010111': 0.0005100071430207,
                                   '1010011000': 0.0005100071430206,
                                   '1010011001': 0.0016925632953668,
                                   '1010011010': 0.0016925632953657,
                                   '1010011011': 0.0005100071430206,
                                   '1010011100': 0.0005100071430206,
                                   '1010011101': 0.0016925632953668,
                                   '1010011110': 0.0005100071430206,
                                   '1010011111': 0.0016925632953657,
                                   '1010100000': 0.0016925632953657,
                                   '1010100001': 0.0005100071430206,
                                   '1010100010': 0.0016925632953663,
                                   '1010100011': 0.0005100071430206,
                                   '1010100100': 0.0005100071430207,
                                   '1010100101': 0.0016925632953657,
                                   '1010100110': 0.0016925632953663,
                                   '1010100111': 0.0005100071430206,
                                   '1010101000': 0.0005100071430207,
                                   '1010101001': 0.0005100071430207,
                                   '1010101010': 0.0016925632953659,
                                   '1010101011': 0.0005100071430207,
                                   '1010101100': 0.0005100071430207,
                                   '1010101101': 0.0005100071430206,
                                   '1010101110': 0.0016925632953663,
                                   '1010101111': 0.0016925632953657,
                                   '1010110000': 0.0016925632953657,
                                   '1010110001': 0.0005100071430206,
                                   '1010110010': 0.0005100071430206,
                                   '1010110011': 0.0016925632953668,
                                   '1010110100': 0.0005100071430206,
                                   '1010110101': 0.0016925632953657,
                                   '1010110110': 0.0005100071430206,
                                   '1010110111': 0.0016925632953668,
                                   '1010111000': 0.0005100071430206,
                                   '1010111001': 0.0005100071430206,
                                   '1010111010': 0.0016925632953657,
                                   '1010111011': 0.0016925632953668,
                                   '1010111100': 0.0005100071430207,
                                   '1010111101': 0.0005100071430207,
                                   '1010111110': 0.0005100071430207,
                                   '1010111111': 0.001692563295366,
                                   '1011000000': 0.0005100071430207,
                                   '1011000001': 0.0005100071430207,
                                   '1011000010': 0.0005100071430207,
                                   '1011000011': 0.0005100071430207,
                                   '1011000100': 0.0028751194477114,
                                   '1011000101': 0.0016925632953659,
                                   '1011000110': 0.0005100071430207,
                                   '1011000111': 0.0005100071430207,
                                   '1011001000': 0.0028751194477113,
                                   '1011001001': 0.0005100071430207,
                                   '1011001010': 0.0016925632953659,
                                   '1011001011': 0.0005100071430207,
                                   '1011001100': 0.0028751194477114,
                                   '1011001101': 0.0005100071430207,
                                   '1011001110': 0.0005100071430207,
                                   '1011001111': 0.0016925632953659,
                                   '1011010000': 0.0016925632953659,
                                   '1011010001': 0.0028751194477114,
                                   '1011010010': 0.0005100071430207,
                                   '1011010011': 0.0005100071430207,
                                   '1011010100': 0.0005100071430207,
                                   '1011010101': 0.0005100071430207,
                                   '1011010110': 0.0005100071430207,
                                   '1011010111': 0.0005100071430207,
                                   '1011011000': 0.0005100071430207,
                                   '1011011001': 0.0028751194477114,
                                   '1011011010': 0.0016925632953659,
                                   '1011011011': 0.0005100071430207,
                                   '1011011100': 0.0005100071430207,
                                   '1011011101': 0.0028751194477113,
                                   '1011011110': 0.0005100071430207,
                                   '1011011111': 0.0016925632953659,
                                   '1011100000': 0.0016925632953659,
                                   '1011100001': 0.0005100071430207,
                                   '1011100010': 0.0028751194477113,
                                   '1011100011': 0.0005100071430207,
                                   '1011100100': 0.0005100071430207,
                                   '1011100101': 0.0016925632953659,
                                   '1011100110': 0.0028751194477114,
                                   '1011100111': 0.0005100071430207,
                                   '1011101000': 0.0005100071430207,
                                   '1011101001': 0.0005100071430207,
                                   '1011101010': 0.0005100071430207,
                                   '1011101011': 0.0005100071430207,
                                   '1011101100': 0.0005100071430207,
                                   '1011101101': 0.0005100071430207,
                                   '1011101110': 0.0028751194477114,
                                   '1011101111': 0.0016925632953659,
                                   '1011110000': 0.0016925632953661,
                                   '1011110001': 0.0005100071430207,
                                   '1011110010': 0.0005100071430207,
                                   '1011110011': 0.0005100071430207,
                                   '1011110100': 0.0005100071430207,
                                   '1011110101': 0.0016925632953661,
                                   '1011110110': 0.0005100071430207,
                                   '1011110111': 0.0005100071430207,
                                   '1011111000': 0.0005100071430207,
                                   '1011111001': 0.0005100071430207,
                                   '1011111010': 0.0016925632953661,
                                   '1011111011': 0.0005100071430207,
                                   '1011111100': 0.0005100071430207,
                                   '1011111101': 0.0005100071430207,
                                   '1011111110': 0.0005100071430207,
                                   '1011111111': 0.0005100071430207,
                                   '1100000000': 0.0016925632953664,
                                   '1100000001': 0.0005100071430206,
                                   '1100000010': 0.0005100071430206,
                                   '1100000011': 0.0005100071430206,
                                   '1100000100': 0.0005100071430207,
                                   '1100000101': 0.0005100071430207,
                                   '1100000110': 0.0005100071430206,
                                   '1100000111': 0.0005100071430206,
                                   '1100001000': 0.0005100071430206,
                                   '1100001001': 0.0005100071430206,
                                   '1100001010': 0.0005100071430207,
                                   '1100001011': 0.0005100071430206,
                                   '1100001100': 0.0005100071430207,
                                   '1100001101': 0.0005100071430206,
                                   '1100001110': 0.0005100071430206,
                                   '1100001111': 0.0005100071430207,
                                   '1100010000': 0.0005100071430207,
                                   '1100010001': 0.0005100071430207,
                                   '1100010010': 0.0005100071430206,
                                   '1100010011': 0.0005100071430206,
                                   '1100010100': 0.0005100071430206,
                                   '1100010101': 0.0016925632953665,
                                   '1100010110': 0.0005100071430206,
                                   '1100010111': 0.0005100071430206,
                                   '1100011000': 0.0005100071430206,
                                   '1100011001': 0.0005100071430207,
                                   '1100011010': 0.0005100071430207,
                                   '1100011011': 0.0005100071430206,
                                   '1100011100': 0.0005100071430206,
                                   '1100011101': 0.0005100071430207,
                                   '1100011110': 0.0005100071430206,
                                   '1100011111': 0.0005100071430207,
                                   '1100100000': 0.0005100071430207,
                                   '1100100001': 0.0005100071430206,
                                   '1100100010': 0.0005100071430207,
                                   '1100100011': 0.0005100071430206,
                                   '1100100100': 0.0005100071430206,
                                   '1100100101': 0.0005100071430207,
                                   '1100100110': 0.0005100071430207,
                                   '1100100111': 0.0005100071430206,
                                   '1100101000': 0.0005100071430206,
                                   '1100101001': 0.0005100071430206,
                                   '1100101010': 0.0016925632953665,
                                   '1100101011': 0.0005100071430206,
                                   '1100101100': 0.0005100071430206,
                                   '1100101101': 0.0005100071430206,
                                   '1100101110': 0.0005100071430207,
                                   '1100101111': 0.0005100071430207,
                                   '1100110000': 0.0005100071430207,
                                   '1100110001': 0.0005100071430206,
                                   '1100110010': 0.0005100071430206,
                                   '1100110011': 0.0005100071430207,
                                   '1100110100': 0.0005100071430206,
                                   '1100110101': 0.0005100071430207,
                                   '1100110110': 0.0005100071430206,
                                   '1100110111': 0.0005100071430207,
                                   '1100111000': 0.0005100071430206,
                                   '1100111001': 0.0005100071430206,
                                   '1100111010': 0.0005100071430207,
                                   '1100111011': 0.0005100071430207,
                                   '1100111100': 0.0005100071430206,
                                   '1100111101': 0.0005100071430206,
                                   '1100111110': 0.0005100071430206,
                                   '1100111111': 0.0016925632953665,
                                   '1101000000': 0.0005100071430207,
                                   '1101000001': 0.0005100071430207,
                                   '1101000010': 0.0005100071430207,
                                   '1101000011': 0.0005100071430207,
                                   '1101000100': 0.001692563295366,
                                   '1101000101': 0.0005100071430207,
                                   '1101000110': 0.0005100071430206,
                                   '1101000111': 0.0005100071430206,
                                   '1101001000': 0.001692563295366,
                                   '1101001001': 0.0005100071430206,
                                   '1101001010': 0.0005100071430207,
                                   '1101001011': 0.0005100071430206,
                                   '1101001100': 0.001692563295366,
                                   '1101001101': 0.0005100071430206,
                                   '1101001110': 0.0005100071430206,
                                   '1101001111': 0.0005100071430207,
                                   '1101010000': 0.0005100071430207,
                                   '1101010001': 0.001692563295366,
                                   '1101010010': 0.0005100071430206,
                                   '1101010011': 0.0005100071430206,
                                   '1101010100': 0.0005100071430207,
                                   '1101010101': 0.0005100071430206,
                                   '1101010110': 0.0005100071430207,
                                   '1101010111': 0.0005100071430207,
                                   '1101011000': 0.0005100071430206,
                                   '1101011001': 0.001692563295366,
                                   '1101011010': 0.0005100071430207,
                                   '1101011011': 0.0005100071430206,
                                   '1101011100': 0.0005100071430206,
                                   '1101011101': 0.0016925632953659,
                                   '1101011110': 0.0005100071430206,
                                   '1101011111': 0.0005100071430207,
                                   '1101100000': 0.0005100071430207,
                                   '1101100001': 0.0005100071430206,
                                   '1101100010': 0.001692563295366,
                                   '1101100011': 0.0005100071430206,
                                   '1101100100': 0.0005100071430206,
                                   '1101100101': 0.0005100071430207,
                                   '1101100110': 0.001692563295366,
                                   '1101100111': 0.0005100071430206,
                                   '1101101000': 0.0005100071430207,
                                   '1101101001': 0.0005100071430207,
                                   '1101101010': 0.0005100071430207,
                                   '1101101011': 0.0005100071430207,
                                   '1101101100': 0.0005100071430206,
                                   '1101101101': 0.0005100071430206,
                                   '1101101110': 0.001692563295366,
                                   '1101101111': 0.0005100071430207,
                                   '1101110000': 0.0005100071430207,
                                   '1101110001': 0.0005100071430206,
                                   '1101110010': 0.0005100071430206,
                                   '1101110011': 0.001692563295366,
                                   '1101110100': 0.0005100071430206,
                                   '1101110101': 0.0005100071430207,
                                   '1101110110': 0.0005100071430206,
                                   '1101110111': 0.001692563295366,
                                   '1101111000': 0.0005100071430206,
                                   '1101111001': 0.0005100071430206,
                                   '1101111010': 0.0005100071430207,
                                   '1101111011': 0.001692563295366,
                                   '1101111100': 0.0005100071430206,
                                   '1101111101': 0.0005100071430206,
                                   '1101111110': 0.0005100071430206,
                                   '1101111111': 0.0005100071430207,
                                   '1110000000': 0.0005100071430207,
                                   '1110000001': 0.0005100071430206,
                                   '1110000010': 0.0005100071430206,
                                   '1110000011': 0.0005100071430206,
                                   '1110000100': 0.0005100071430207,
                                   '1110000101': 0.0016925632953659,
                                   '1110000110': 0.0005100071430207,
                                   '1110000111': 0.0005100071430207,
                                   '1110001000': 0.0005100071430207,
                                   '1110001001': 0.0005100071430207,
                                   '1110001010': 0.0016925632953659,
                                   '1110001011': 0.0005100071430207,
                                   '1110001100': 0.0005100071430207,
                                   '1110001101': 0.0005100071430207,
                                   '1110001110': 0.0005100071430207,
                                   '1110001111': 0.0016925632953659,
                                   '1110010000': 0.0016925632953659,
                                   '1110010001': 0.0005100071430207,
                                   '1110010010': 0.0005100071430207,
                                   '1110010011': 0.0005100071430207,
                                   '1110010100': 0.0005100071430206,
                                   '1110010101': 0.0005100071430207,
                                   '1110010110': 0.0005100071430206,
                                   '1110010111': 0.0005100071430206,
                                   '1110011000': 0.0005100071430207,
                                   '1110011001': 0.0005100071430207,
                                   '1110011010': 0.0016925632953659,
                                   '1110011011': 0.0005100071430207,
                                   '1110011100': 0.0005100071430207,
                                   '1110011101': 0.0005100071430207,
                                   '1110011110': 0.0005100071430207,
                                   '1110011111': 0.0016925632953659,
                                   '1110100000': 0.0016925632953659,
                                   '1110100001': 0.0005100071430207,
                                   '1110100010': 0.0005100071430207,
                                   '1110100011': 0.0005100071430207,
                                   '1110100100': 0.0005100071430207,
                                   '1110100101': 0.0016925632953659,
                                   '1110100110': 0.0005100071430207,
                                   '1110100111': 0.0005100071430207,
                                   '1110101000': 0.0005100071430207,
                                   '1110101001': 0.0005100071430207,
                                   '1110101010': 0.0005100071430207,
                                   '1110101011': 0.0005100071430207,
                                   '1110101100': 0.0005100071430207,
                                   '1110101101': 0.0005100071430207,
                                   '1110101110': 0.0005100071430207,
                                   '1110101111': 0.0016925632953659,
                                   '1110110000': 0.0016925632953659,
                                   '1110110001': 0.0005100071430207,
                                   '1110110010': 0.0005100071430207,
                                   '1110110011': 0.0005100071430207,
                                   '1110110100': 0.0005100071430207,
                                   '1110110101': 0.0016925632953659,
                                   '1110110110': 0.0005100071430207,
                                   '1110110111': 0.0005100071430207,
                                   '1110111000': 0.0005100071430207,
                                   '1110111001': 0.0005100071430207,
                                   '1110111010': 0.0016925632953659,
                                   '1110111011': 0.0005100071430207,
                                   '1110111100': 0.0005100071430206,
                                   '1110111101': 0.0005100071430206,
                                   '1110111110': 0.0005100071430206,
                                   '1110111111': 0.0005100071430207,
                                   '1111000000': 0.001692563295366,
                                   '1111000001': 0.0005100071430207,
                                   '1111000010': 0.0005100071430207,
                                   '1111000011': 0.0005100071430207,
                                   '1111000100': 0.0016925632953657,
                                   '1111000101': 0.0016925632953656,
                                   '1111000110': 0.0005100071430206,
                                   '1111000111': 0.0005100071430206,
                                   '1111001000': 0.0016925632953657,
                                   '1111001001': 0.0005100071430206,
                                   '1111001010': 0.0016925632953656,
                                   '1111001011': 0.0005100071430206,
                                   '1111001100': 0.0016925632953657,
                                   '1111001101': 0.0005100071430206,
                                   '1111001110': 0.0005100071430206,
                                   '1111001111': 0.0016925632953656,
                                   '1111010000': 0.0016925632953656,
                                   '1111010001': 0.0016925632953657,
                                   '1111010010': 0.0005100071430206,
                                   '1111010011': 0.0005100071430206,
                                   '1111010100': 0.0005100071430207,
                                   '1111010101': 0.001692563295366,
                                   '1111010110': 0.0005100071430207,
                                   '1111010111': 0.0005100071430207,
                                   '1111011000': 0.0005100071430206,
                                   '1111011001': 0.0016925632953657,
                                   '1111011010': 0.0016925632953656,
                                   '1111011011': 0.0005100071430206,
                                   '1111011100': 0.0005100071430206,
                                   '1111011101': 0.0016925632953657,
                                   '1111011110': 0.0005100071430206,
                                   '1111011111': 0.0016925632953656,
                                   '1111100000': 0.0016925632953656,
                                   '1111100001': 0.0005100071430206,
                                   '1111100010': 0.0016925632953657,
                                   '1111100011': 0.0005100071430206,
                                   '1111100100': 0.0005100071430206,
                                   '1111100101': 0.0016925632953656,
                                   '1111100110': 0.0016925632953657,
                                   '1111100111': 0.0005100071430206,
                                   '1111101000': 0.0005100071430207,
                                   '1111101001': 0.0005100071430207,
                                   '1111101010': 0.001692563295366,
                                   '1111101011': 0.0005100071430207,
                                   '1111101100': 0.0005100071430206,
                                   '1111101101': 0.0005100071430206,
                                   '1111101110': 0.0016925632953657,
                                   '1111101111': 0.0016925632953656,
                                   '1111110000': 0.0016925632953656,
                                   '1111110001': 0.0005100071430207,
                                   '1111110010': 0.0005100071430207,
                                   '1111110011': 0.0016925632953657,
                                   '1111110100': 0.0005100071430207,
                                   '1111110101': 0.0016925632953656,
                                   '1111110110': 0.0005100071430207,
                                   '1111110111': 0.0016925632953657,
                                   '1111111000': 0.0005100071430207,
                                   '1111111001': 0.0005100071430207,
                                   '1111111010': 0.0016925632953656,
                                   '1111111011': 0.0016925632953657,
                                   '1111111100': 0.0005100071430207,
                                   '1111111101': 0.0005100071430207,
                                   '1111111110': 0.0005100071430207,
                                   '1111111111': 0.0016925632953659}],
        'iterations': [1],
        'max_probability': 0.0028751194477114,
        'oracle_evaluation': True,
        'top_measurement': '0001000110'}
    Plotting for bitstring 0001000110 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'green', 4: 'blue'}
    Plotting for bitstring 0001000111 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'green', 4: 'yellow'}
    Plotting for bitstring 0001001001 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'blue', 4: 'green'}
    Plotting for bitstring 0001001011 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'blue', 4: 'yellow'}
    Plotting for bitstring 0001001101 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'yellow', 4: 'green'}
    Plotting for bitstring 0001001110 with color assignment: {0: 'red', 1: 'green', 2: 'red', 3: 'yellow', 4: 'blue'}
    Plotting for bitstring 0001100001 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'red', 4: 'green'}
    Plotting for bitstring 0001100011 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'red', 4: 'yellow'}
    Plotting for bitstring 0001100100 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'green', 4: 'red'}
    Plotting for bitstring 0001100111 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'green', 4: 'yellow'}
    Plotting for bitstring 0001101100 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'red'}
    Plotting for bitstring 0001101101 with color assignment: {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'green'}
    Plotting for bitstring 0001110001 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'red', 4: 'green'}
    Plotting for bitstring 0001110010 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'red', 4: 'blue'}
    Plotting for bitstring 0001110100 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'green', 4: 'red'}
    Plotting for bitstring 0001110110 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'green', 4: 'blue'}
    Plotting for bitstring 0001111000 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'red'}
    Plotting for bitstring 0001111001 with color assignment: {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'green'}
    Plotting for bitstring 1001000100 with color assignment: {0: 'blue', 1: 'green', 2: 'red', 3: 'green', 4: 'red'}
    Plotting for bitstring 1001001000 with color assignment: {0: 'blue', 1: 'green', 2: 'red', 3: 'blue', 4: 'red'}



    
![png](presentation_files/presentation_24_3.png)
    


             ┌───┐                                                        »
       q1_0: ┤ H ├─■──────────────────────────────────────────────────────»
             ├───┤ │                                                      »
       q1_1: ┤ H ├─┼─────────────■────────────────────────────────────────»
             ├───┤ │             │                                        »
       q1_2: ┤ H ├─┼─────────────┼─────────────■──────────────────────────»
             ├───┤ │             │             │                          »
       q1_3: ┤ H ├─┼─────────────┼─────────────┼─────────────■────────────»
             ├───┤ │ZZ(0.22301)  │             │             │            »
       q1_4: ┤ H ├─■─────────────┼─────────────┼─────────────┼────────────»
             ├───┤               │ZZ(0.22301)  │             │            »
       q1_5: ┤ H ├───────────────■─────────────┼─────────────┼────────────»
             ├───┤                             │ZZ(0.22301)  │            »
       q1_6: ┤ H ├─────────────────────────────■─────────────┼────────────»
             ├───┤                                           │ZZ(0.22301) »
       q1_7: ┤ H ├───────────────────────────────────────────■────────────»
             ├───┤                                                        »
       q1_8: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q1_9: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_10: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_11: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_12: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_13: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_14: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_15: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_16: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_17: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_18: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
      q1_19: ┤ H ├────────────────────────────────────────────────────────»
             └───┘                                                        »
    meas: 20/═════════════════════════════════════════════════════════════»
                                                                          »
    «                                                                 »
    «   q1_0: ─■────────────────────────────■─────────────■───────────»
    «          │                            │ZZ(9.032)    │           »
    «   q1_1: ─┼─────────────■──────────────■─────────────┼───────────»
    «          │             │                            │ZZ(9.032)  »
    «   q1_2: ─┼─────────────┼─────────────■──────────────■───────────»
    «          │             │             │                          »
    «   q1_3: ─┼─────────────┼─────────────┼─────────────■────────────»
    «          │             │             │             │            »
    «   q1_4: ─┼─────────────┼─────────────┼─────────────┼────────────»
    «          │             │             │             │            »
    «   q1_5: ─┼─────────────┼─────────────┼─────────────┼────────────»
    «          │             │             │             │            »
    «   q1_6: ─┼─────────────┼─────────────┼─────────────┼────────────»
    «          │             │             │             │            »
    «   q1_7: ─┼─────────────┼─────────────┼─────────────┼────────────»
    «          │ZZ(0.22301)  │             │             │            »
    «   q1_8: ─■─────────────┼─────────────┼─────────────┼────────────»
    «                        │ZZ(0.22301)  │             │            »
    «   q1_9: ───────────────■─────────────┼─────────────┼────────────»
    «                                      │ZZ(0.22301)  │            »
    «  q1_10: ─────────────────────────────■─────────────┼────────────»
    «                                                    │ZZ(0.22301) »
    «  q1_11: ───────────────────────────────────────────■────────────»
    «                                                                 »
    «  q1_12: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_13: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_14: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_15: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_16: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_17: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_18: ────────────────────────────────────────────────────────»
    «                                                                 »
    «  q1_19: ────────────────────────────────────────────────────────»
    «                                                                 »
    «meas: 20/════════════════════════════════════════════════════════»
    «                                                                 »
    «                       ┌────────────┐┌─────────────┐              »
    «   q1_0: ──■───────────┤ Rz(-18.51) ├┤ Rx(-1.2335) ├──────────────»
    «           │           └────────────┘└─────────────┘┌────────────┐»
    «   q1_1: ──┼─────────────■──────────────■───────────┤ Rz(-18.51) ├»
    «           │             │ZZ(9.032)     │           └────────────┘»
    «   q1_2: ──┼─────────────■──────────────┼─────────────■───────────»
    «           │ZZ(9.032)                   │ZZ(9.032)    │ZZ(9.032)  »
    «   q1_3: ──■────────────────────────────■─────────────■───────────»
    «                                                                  »
    «   q1_4: ─■─────────────────────────────■─────────────■───────────»
    «          │                             │ZZ(9.032)    │           »
    «   q1_5: ─┼─────────────■───────────────■─────────────┼───────────»
    «          │             │                             │ZZ(9.032)  »
    «   q1_6: ─┼─────────────┼──────────────■──────────────■───────────»
    «          │             │              │                          »
    «   q1_7: ─┼─────────────┼──────────────┼─────────────■────────────»
    «          │ZZ(0.22301)  │              │             │            »
    «   q1_8: ─■─────────────┼──────────────┼─────────────┼────────────»
    «                        │ZZ(0.22301)   │             │            »
    «   q1_9: ───────────────■──────────────┼─────────────┼────────────»
    «                                       │ZZ(0.22301)  │            »
    «  q1_10: ──────────────────────────────■─────────────┼────────────»
    «                                                     │ZZ(0.22301) »
    «  q1_11: ────────────────────────────────────────────■────────────»
    «                                                                  »
    «  q1_12: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_13: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_14: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_15: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_16: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_17: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_18: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «  q1_19: ─────────────────────────────────────────────────────────»
    «                                                                  »
    «meas: 20/═════════════════════════════════════════════════════════»
    «                                                                  »
    «                                                                    »
    «   q1_0: ──────────────────────────────────────────────■────────────»
    «         ┌─────────────┐                               │            »
    «   q1_1: ┤ Rx(-1.2335) ├───────────────────────────────┼────────────»
    «         └┬────────────┤┌─────────────┐                │            »
    «   q1_2: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├────────────────┼────────────»
    «          ├────────────┤├─────────────┤                │            »
    «   q1_3: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├────────────────┼────────────»
    «          └────────────┘└┬────────────┤┌─────────────┐ │ZZ(0.69749) »
    «   q1_4: ───■────────────┤ Rz(-18.51) ├┤ Rx(-1.2335) ├─■────────────»
    «            │            └────────────┘└─────────────┘┌────────────┐»
    «   q1_5: ───┼──────────────■──────────────■───────────┤ Rz(-18.51) ├»
    «            │              │ZZ(9.032)     │           └────────────┘»
    «   q1_6: ───┼──────────────■──────────────┼─────────────■───────────»
    «            │ZZ(9.032)                    │ZZ(9.032)    │ZZ(9.032)  »
    «   q1_7: ───■─────────────────────────────■─────────────■───────────»
    «                                                                    »
    «   q1_8: ──■──────────────────────────────■─────────────■───────────»
    «           │                              │ZZ(9.032)    │           »
    «   q1_9: ──┼──────────────■───────────────■─────────────┼───────────»
    «           │              │                             │ZZ(9.032)  »
    «  q1_10: ──┼──────────────┼──────────────■──────────────■───────────»
    «           │              │              │                          »
    «  q1_11: ──┼──────────────┼──────────────┼─────────────■────────────»
    «           │ZZ(0.22301)   │              │             │            »
    «  q1_12: ──■──────────────┼──────────────┼─────────────┼────────────»
    «                          │ZZ(0.22301)   │             │            »
    «  q1_13: ─────────────────■──────────────┼─────────────┼────────────»
    «                                         │ZZ(0.22301)  │            »
    «  q1_14: ────────────────────────────────■─────────────┼────────────»
    «                                                       │ZZ(0.22301) »
    «  q1_15: ──────────────────────────────────────────────■────────────»
    «                                                                    »
    «  q1_16: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «  q1_17: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «  q1_18: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «  q1_19: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «meas: 20/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                                     »
    «   q1_0: ───────────────────────────────────────────────■────────────»
    «                                                        │            »
    «   q1_1: ─────────────────■─────────────────────────────┼────────────»
    «                          │                             │            »
    «   q1_2: ─────────────────┼──────────────■──────────────┼────────────»
    «                          │              │              │            »
    «   q1_3: ─────────────────┼──────────────┼──────────────┼────────────»
    «                          │              │              │            »
    «   q1_4: ─────────────────┼──────────────┼──────────────┼────────────»
    «         ┌─────────────┐  │ZZ(0.69749)   │              │            »
    «   q1_5: ┤ Rx(-1.2335) ├──■──────────────┼──────────────┼────────────»
    «         └┬────────────┤┌─────────────┐  │ZZ(0.69749)   │            »
    «   q1_6: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├──■──────────────┼────────────»
    «          ├────────────┤├─────────────┤                 │            »
    «   q1_7: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├─────────────────┼────────────»
    «          └────────────┘├─────────────┤┌─────────────┐  │ZZ(0.69749) »
    «   q1_8: ───■───────────┤ Rz(-18.733) ├┤ Rx(-1.2335) ├──■────────────»
    «            │           └─────────────┘└─────────────┘┌─────────────┐»
    «   q1_9: ───┼──────────────■──────────────■───────────┤ Rz(-18.733) ├»
    «            │              │ZZ(9.032)     │           └─────────────┘»
    «  q1_10: ───┼──────────────■──────────────┼──────────────■───────────»
    «            │ZZ(9.032)                    │ZZ(9.032)     │ZZ(9.032)  »
    «  q1_11: ───■─────────────────────────────■──────────────■───────────»
    «                                                                     »
    «  q1_12: ──■──────────────────────────────■──────────────■───────────»
    «           │                              │ZZ(9.032)     │           »
    «  q1_13: ──┼──────────────■───────────────■──────────────┼───────────»
    «           │              │                              │ZZ(9.032)  »
    «  q1_14: ──┼──────────────┼──────────────■───────────────■───────────»
    «           │              │              │                           »
    «  q1_15: ──┼──────────────┼──────────────┼──────────────■────────────»
    «           │ZZ(0.22301)   │              │              │            »
    «  q1_16: ──■──────────────┼──────────────┼──────────────┼────────────»
    «                          │ZZ(0.22301)   │              │            »
    «  q1_17: ─────────────────■──────────────┼──────────────┼────────────»
    «                                         │ZZ(0.22301)   │            »
    «  q1_18: ────────────────────────────────■──────────────┼────────────»
    «                                                        │ZZ(0.22301) »
    «  q1_19: ───────────────────────────────────────────────■────────────»
    «                                                                     »
    «meas: 20/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                                                     »
    «   q1_0: ────────────────────────────────■───────────────────────────»
    «                                         │ZZ(28.249)                 »
    «   q1_1: ─────────────────■──────────────■───────────────────────────»
    «                          │                                          »
    «   q1_2: ─────────────────┼─────────────────────────────■────────────»
    «                          │                             │            »
    «   q1_3: ──■──────────────┼─────────────────────────────┼────────────»
    «           │              │                             │            »
    «   q1_4: ──┼──────────────┼──────────────■──────────────┼────────────»
    «           │              │              │              │            »
    «   q1_5: ──┼──────────────┼──────────────┼──────────────┼────────────»
    «           │              │              │              │            »
    «   q1_6: ──┼──────────────┼──────────────┼──────────────┼────────────»
    «           │ZZ(0.69749)   │              │              │            »
    «   q1_7: ──■──────────────┼──────────────┼──────────────┼────────────»
    «                          │              │ZZ(0.69749)   │            »
    «   q1_8: ─────────────────┼──────────────■──────────────┼────────────»
    «         ┌─────────────┐  │ZZ(0.69749)                  │            »
    «   q1_9: ┤ Rx(-1.2335) ├──■─────────────────────────────┼────────────»
    «         ├─────────────┤┌─────────────┐                 │ZZ(0.69749) »
    «  q1_10: ┤ Rz(-18.733) ├┤ Rx(-1.2335) ├─────────────────■────────────»
    «         ├─────────────┤├─────────────┤                              »
    «  q1_11: ┤ Rz(-18.733) ├┤ Rx(-1.2335) ├──────────────────────────────»
    «         └─────────────┘└┬────────────┤┌─────────────┐               »
    «  q1_12: ───■────────────┤ Rz(-18.51) ├┤ Rx(-1.2335) ├───────────────»
    «            │            └────────────┘└─────────────┘ ┌────────────┐»
    «  q1_13: ───┼──────────────■──────────────■────────────┤ Rz(-18.51) ├»
    «            │              │ZZ(9.032)     │            └────────────┘»
    «  q1_14: ───┼──────────────■──────────────┼──────────────■───────────»
    «            │ZZ(9.032)                    │ZZ(9.032)     │ZZ(9.032)  »
    «  q1_15: ───■─────────────────────────────■──────────────■───────────»
    «                                                      ┌─────────────┐»
    «  q1_16: ───■──────────────■──────────────■───────────┤ Rz(-18.287) ├»
    «            │ZZ(9.032)     │              │           └─────────────┘»
    «  q1_17: ───■──────────────┼──────────────┼──────────────■───────────»
    «                           │ZZ(9.032)     │              │ZZ(9.032)  »
    «  q1_18: ──────────────────■──────────────┼──────────────■───────────»
    «                                          │ZZ(9.032)                 »
    «  q1_19: ─────────────────────────────────■──────────────────────────»
    «                                                                     »
    «meas: 20/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                       ┌─────────────┐┌──────────────┐»
    «   q1_0: ──■──────────────■────────────┤ Rz(-57.892) ├┤ Rx(-0.71884) ├»
    «           │              │            └─────────────┘└──────────────┘»
    «   q1_1: ──┼──────────────┼──────────────■──────────────■─────────────»
    «           │ZZ(28.249)    │              │ZZ(28.249)    │             »
    «   q1_2: ──■──────────────┼──────────────■──────────────┼─────────────»
    «                          │ZZ(28.249)                   │ZZ(28.249)   »
    «   q1_3: ──■──────────────■─────────────────────────────■─────────────»
    «           │                                                          »
    «   q1_4: ──┼─────────────────────────────■────────────────────────────»
    «           │                             │ZZ(28.249)                  »
    «   q1_5: ──┼──────────────■──────────────■────────────────────────────»
    «           │              │                                           »
    «   q1_6: ──┼──────────────┼─────────────────────────────■─────────────»
    «           │              │                             │             »
    «   q1_7: ──┼──────────────┼─────────────────────────────┼─────────────»
    «           │              │                             │             »
    «   q1_8: ──┼──────────────┼──────────────■──────────────┼─────────────»
    «           │              │ZZ(0.69749)   │              │             »
    «   q1_9: ──┼──────────────■──────────────┼──────────────┼─────────────»
    «           │                             │              │ZZ(0.69749)  »
    «  q1_10: ──┼─────────────────────────────┼──────────────■─────────────»
    «           │ZZ(0.69749)                  │                            »
    «  q1_11: ──■─────────────────────────────┼────────────────────────────»
    «                                         │ZZ(0.69749)                 »
    «  q1_12: ────────────────────────────────■──────────────■─────────────»
    «         ┌─────────────┐                                │             »
    «  q1_13: ┤ Rx(-1.2335) ├────────────────────────────────┼─────────────»
    «         └┬────────────┤┌─────────────┐                 │             »
    «  q1_14: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├─────────────────┼─────────────»
    «          ├────────────┤├─────────────┤                 │             »
    «  q1_15: ─┤ Rz(-18.51) ├┤ Rx(-1.2335) ├─────────────────┼─────────────»
    «         ┌┴────────────┤└─────────────┘                 │ZZ(0.69749)  »
    «  q1_16: ┤ Rx(-1.2335) ├────────────────────────────────■─────────────»
    «         └─────────────┘┌─────────────┐┌─────────────┐                »
    «  q1_17: ───■───────────┤ Rz(-18.287) ├┤ Rx(-1.2335) ├────────────────»
    «            │           └─────────────┘├─────────────┤┌─────────────┐ »
    «  q1_18: ───┼──────────────■───────────┤ Rz(-18.287) ├┤ Rx(-1.2335) ├─»
    «            │ZZ(9.032)     │ZZ(9.032)  ├─────────────┤├─────────────┤ »
    «  q1_19: ───■──────────────■───────────┤ Rz(-18.287) ├┤ Rx(-1.2335) ├─»
    «                                       └─────────────┘└─────────────┘ »
    «meas: 20/═════════════════════════════════════════════════════════════»
    «                                                                      »
    «                                                                        »
    «   q1_0: ───────────────────────────────────────────────────────────────»
    «         ┌─────────────┐┌──────────────┐                                »
    «   q1_1: ┤ Rz(-57.892) ├┤ Rx(-0.71884) ├────────────────────────────────»
    «         └─────────────┘├─────────────┬┘┌──────────────┐                »
    «   q1_2: ──■────────────┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├────────────────»
    «           │ZZ(28.249)  ├─────────────┤ ├──────────────┤                »
    «   q1_3: ──■────────────┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├────────────────»
    «                        └─────────────┘ ├─────────────┬┘┌──────────────┐»
    «   q1_4: ──■──────────────■─────────────┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├»
    «           │              │             └─────────────┘ └──────────────┘»
    «   q1_5: ──┼──────────────┼───────────────■───────────────■─────────────»
    «           │ZZ(28.249)    │               │ZZ(28.249)     │             »
    «   q1_6: ──■──────────────┼───────────────■───────────────┼─────────────»
    «                          │ZZ(28.249)                     │ZZ(28.249)   »
    «   q1_7: ──■──────────────■───────────────────────────────■─────────────»
    «           │                                                            »
    «   q1_8: ──┼──────────────────────────────■───────────────■─────────────»
    «           │                              │ZZ(28.249)     │             »
    «   q1_9: ──┼──────────────■───────────────■───────────────┼─────────────»
    «           │              │                               │ZZ(28.249)   »
    «  q1_10: ──┼──────────────┼───────────────■───────────────■─────────────»
    «           │ZZ(0.69749)   │               │                             »
    «  q1_11: ──■──────────────┼───────────────┼───────────────■─────────────»
    «                          │               │               │             »
    «  q1_12: ─────────────────┼───────────────┼───────────────┼─────────────»
    «                          │ZZ(0.69749)    │               │             »
    «  q1_13: ─────────────────■───────────────┼───────────────┼─────────────»
    «                                          │ZZ(0.69749)    │             »
    «  q1_14: ─────────────────────────────────■───────────────┼─────────────»
    «                                                          │ZZ(0.69749)  »
    «  q1_15: ─────────────────────────────────────────────────■─────────────»
    «                                                                        »
    «  q1_16: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_17: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_18: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_19: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «meas: 20/═══════════════════════════════════════════════════════════════»
    «                                                                        »
    «                                                                       »
    «   q1_0: ──■──────────────────────────────────────────────■────────────»
    «           │                                              │            »
    «   q1_1: ──┼──────────────────────────────■───────────────┼────────────»
    «           │                              │               │            »
    «   q1_2: ──┼──────────────────────────────┼───────────────┼────────────»
    «           │                              │               │            »
    «   q1_3: ──┼──────────────────────────────┼───────────────┼────────────»
    «           │ZZ(1.1361)                    │               │            »
    «   q1_4: ──■──────────────────────────────┼───────────────┼────────────»
    «         ┌─────────────┐┌──────────────┐  │ZZ(1.1361)     │            »
    «   q1_5: ┤ Rz(-57.892) ├┤ Rx(-0.71884) ├──■───────────────┼────────────»
    «         └─────────────┘├─────────────┬┘┌──────────────┐  │            »
    «   q1_6: ──■────────────┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├──┼────────────»
    «           │ZZ(28.249)  ├─────────────┤ ├──────────────┤  │            »
    «   q1_7: ──■────────────┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├──┼────────────»
    «                        ├─────────────┤ ├──────────────┤  │ZZ(1.1361)  »
    «   q1_8: ──■────────────┤ Rz(-58.589) ├─┤ Rx(-0.71884) ├──■────────────»
    «           │            └─────────────┘ └──────────────┘┌─────────────┐»
    «   q1_9: ──┼──────────────■───────────────■─────────────┤ Rz(-58.589) ├»
    «           │              │ZZ(28.249)     │             └─────────────┘»
    «  q1_10: ──┼──────────────■───────────────┼───────────────■────────────»
    «           │ZZ(28.249)                    │ZZ(28.249)     │ZZ(28.249)  »
    «  q1_11: ──■──────────────────────────────■───────────────■────────────»
    «                                                                       »
    «  q1_12: ─────────────────■───────────────■───────────────■────────────»
    «                          │ZZ(28.249)     │               │            »
    «  q1_13: ──■──────────────■───────────────┼───────────────┼────────────»
    «           │                              │ZZ(28.249)     │            »
    «  q1_14: ──┼──────────────■───────────────■───────────────┼────────────»
    «           │              │                               │ZZ(28.249)  »
    «  q1_15: ──┼──────────────┼───────────────■───────────────■────────────»
    «           │              │               │                            »
    «  q1_16: ──┼──────────────┼───────────────┼───────────────■────────────»
    «           │ZZ(0.69749)   │               │               │ZZ(28.249)  »
    «  q1_17: ──■──────────────┼───────────────┼───────────────■────────────»
    «                          │ZZ(0.69749)    │                            »
    «  q1_18: ─────────────────■───────────────┼────────────────────────────»
    «                                          │ZZ(0.69749)                 »
    «  q1_19: ─────────────────────────────────■────────────────────────────»
    «                                                                       »
    «meas: 20/══════════════════════════════════════════════════════════════»
    «                                                                       »
    «                                                                        »
    «   q1_0: ──────────────────────────────────────────────────■────────────»
    «                                                           │ZZ(46.01)   »
    «   q1_1: ──────────────────────────────────■───────────────■────────────»
    «                                           │                            »
    «   q1_2: ──■───────────────────────────────┼────────────────────────────»
    «           │                               │                            »
    «   q1_3: ──┼───────────────■───────────────┼────────────────────────────»
    «           │               │               │                            »
    «   q1_4: ──┼───────────────┼───────────────┼──────────────■─────────────»
    «           │               │               │              │             »
    «   q1_5: ──┼───────────────┼───────────────┼──────────────┼─────────────»
    «           │ZZ(1.1361)     │               │              │             »
    «   q1_6: ──■───────────────┼───────────────┼──────────────┼─────────────»
    «                           │ZZ(1.1361)     │              │             »
    «   q1_7: ──────────────────■───────────────┼──────────────┼─────────────»
    «                                           │              │ZZ(1.1361)   »
    «   q1_8: ──────────────────────────────────┼──────────────■─────────────»
    «         ┌──────────────┐                  │ZZ(1.1361)                  »
    «   q1_9: ┤ Rx(-0.71884) ├──────────────────■────────────────────────────»
    «         ├─────────────┬┘┌──────────────┐                               »
    «  q1_10: ┤ Rz(-58.589) ├─┤ Rx(-0.71884) ├───────────────────────────────»
    «         ├─────────────┤ ├──────────────┤                               »
    «  q1_11: ┤ Rz(-58.589) ├─┤ Rx(-0.71884) ├───────────────────────────────»
    «         ├─────────────┤ ├──────────────┤                               »
    «  q1_12: ┤ Rz(-57.892) ├─┤ Rx(-0.71884) ├───────────────────────────────»
    «         └─────────────┘ └──────────────┘┌─────────────┐┌──────────────┐»
    «  q1_13: ──■───────────────■─────────────┤ Rz(-57.892) ├┤ Rx(-0.71884) ├»
    «           │ZZ(28.249)     │             └─────────────┘├─────────────┬┘»
    «  q1_14: ──■───────────────┼───────────────■────────────┤ Rz(-57.892) ├─»
    «                           │ZZ(28.249)     │ZZ(28.249)  ├─────────────┤ »
    «  q1_15: ──────────────────■───────────────■────────────┤ Rz(-57.892) ├─»
    «                                         ┌─────────────┐├─────────────┴┐»
    «  q1_16: ──■───────────────■─────────────┤ Rz(-57.195) ├┤ Rx(-0.71884) ├»
    «           │               │             └─────────────┘└──────────────┘»
    «  q1_17: ──┼───────────────┼───────────────■──────────────■─────────────»
    «           │ZZ(28.249)     │               │ZZ(28.249)    │             »
    «  q1_18: ──■───────────────┼───────────────■──────────────┼─────────────»
    «                           │ZZ(28.249)                    │ZZ(28.249)   »
    «  q1_19: ──────────────────■──────────────────────────────■─────────────»
    «                                                                        »
    «meas: 20/═══════════════════════════════════════════════════════════════»
    «                                                                        »
    «                                                         ┌─────────────┐»
    «   q1_0: ───────────────────■───────────────■────────────┤ Rz(-94.292) ├»
    «                            │               │            └─────────────┘»
    «   q1_1: ───────────────────┼───────────────┼───────────────■───────────»
    «                            │ZZ(46.01)      │               │ZZ(46.01)  »
    «   q1_2: ──■────────────────■───────────────┼───────────────■───────────»
    «           │                                │ZZ(46.01)                  »
    «   q1_3: ──┼───────────────■────────────────■───────────────────────────»
    «           │               │                                            »
    «   q1_4: ──┼───────────────┼────────────────────────────────■───────────»
    «           │               │                                │ZZ(46.01)  »
    «   q1_5: ──┼───────────────┼───────────────■────────────────■───────────»
    «           │               │               │                            »
    «   q1_6: ──┼───────────────┼───────────────┼────────────────────────────»
    «           │               │               │                            »
    «   q1_7: ──┼───────────────┼───────────────┼────────────────────────────»
    «           │               │               │                            »
    «   q1_8: ──┼───────────────┼───────────────┼───────────────■────────────»
    «           │               │               │ZZ(1.1361)     │            »
    «   q1_9: ──┼───────────────┼───────────────■───────────────┼────────────»
    «           │ZZ(1.1361)     │                               │            »
    «  q1_10: ──■───────────────┼───────────────────────────────┼────────────»
    «                           │ZZ(1.1361)                     │            »
    «  q1_11: ──────────────────■───────────────────────────────┼────────────»
    «                                                           │ZZ(1.1361)  »
    «  q1_12: ──────────────────────────────────────────────────■────────────»
    «                                                                        »
    «  q1_13: ───────────────────────────────────────────────────────────────»
    «         ┌──────────────┐                                               »
    «  q1_14: ┤ Rx(-0.71884) ├───────────────────────────────────────────────»
    «         ├──────────────┤                                               »
    «  q1_15: ┤ Rx(-0.71884) ├───────────────────────────────────────────────»
    «         └──────────────┘                                               »
    «  q1_16: ───────────────────────────────────────────────────────────────»
    «         ┌─────────────┐ ┌──────────────┐                               »
    «  q1_17: ┤ Rz(-57.195) ├─┤ Rx(-0.71884) ├───────────────────────────────»
    «         └─────────────┘ ├─────────────┬┘┌──────────────┐               »
    «  q1_18: ──■─────────────┤ Rz(-57.195) ├─┤ Rx(-0.71884) ├───────────────»
    «           │ZZ(28.249)   ├─────────────┤ ├──────────────┤               »
    «  q1_19: ──■─────────────┤ Rz(-57.195) ├─┤ Rx(-0.71884) ├───────────────»
    «                         └─────────────┘ └──────────────┘               »
    «meas: 20/═══════════════════════════════════════════════════════════════»
    «                                                                        »
    «         ┌──────────────┐                                               »
    «   q1_0: ┤ Rx(-0.24503) ├───────────────────────────────────────────────»
    «         └──────────────┘┌─────────────┐┌──────────────┐                »
    «   q1_1: ───■────────────┤ Rz(-94.292) ├┤ Rx(-0.24503) ├────────────────»
    «            │            └─────────────┘├─────────────┬┘┌──────────────┐»
    «   q1_2: ───┼───────────────■───────────┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├»
    «            │ZZ(46.01)      │ZZ(46.01)  ├─────────────┤ ├──────────────┤»
    «   q1_3: ───■───────────────■───────────┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├»
    «                                        └─────────────┘ ├─────────────┬┘»
    «   q1_4: ───────────────────■──────────────■────────────┤ Rz(-94.292) ├─»
    «                            │              │            └─────────────┘ »
    «   q1_5: ───────────────────┼──────────────┼───────────────■────────────»
    «                            │ZZ(46.01)     │               │ZZ(46.01)   »
    «   q1_6: ──■────────────────■──────────────┼───────────────■────────────»
    «           │                               │ZZ(46.01)                   »
    «   q1_7: ──┼───────────────■───────────────■────────────────────────────»
    «           │               │                                            »
    «   q1_8: ──┼───────────────┼───────────────────────────────■────────────»
    «           │               │                               │ZZ(46.01)   »
    «   q1_9: ──┼───────────────┼──────────────■────────────────■────────────»
    «           │ZZ(1.1361)     │              │                             »
    «  q1_10: ──■───────────────┼──────────────┼───────────────■─────────────»
    «                           │ZZ(1.1361)    │               │             »
    «  q1_11: ──────────────────■──────────────┼───────────────┼─────────────»
    «                                          │               │             »
    «  q1_12: ──■──────────────────────────────┼───────────────┼─────────────»
    «           │                              │ZZ(1.1361)     │             »
    «  q1_13: ──┼──────────────────────────────■───────────────┼─────────────»
    «           │                                              │ZZ(1.1361)   »
    «  q1_14: ──┼──────────────────────────────────────────────■─────────────»
    «           │                                                            »
    «  q1_15: ──┼────────────────────────────────────────────────────────────»
    «           │ZZ(1.1361)                                                  »
    «  q1_16: ──■────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_17: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_18: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «  q1_19: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «meas: 20/═══════════════════════════════════════════════════════════════»
    «                                                                        »
    «                                                                        »
    «   q1_0: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «   q1_1: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «   q1_2: ───────────────────────────────────────────────────────────────»
    «                                                                        »
    «   q1_3: ───────────────────────────────────────────────────────────────»
    «         ┌──────────────┐                                               »
    «   q1_4: ┤ Rx(-0.24503) ├───────────────────────────────────────────────»
    «         └──────────────┘┌─────────────┐┌──────────────┐                »
    «   q1_5: ───■────────────┤ Rz(-94.292) ├┤ Rx(-0.24503) ├────────────────»
    «            │            └─────────────┘├─────────────┬┘┌──────────────┐»
    «   q1_6: ───┼───────────────■───────────┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├»
    «            │ZZ(46.01)      │ZZ(46.01)  ├─────────────┤ ├──────────────┤»
    «   q1_7: ───■───────────────■───────────┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├»
    «                                        ├─────────────┤ ├──────────────┤»
    «   q1_8: ───■───────────────■───────────┤ Rz(-95.428) ├─┤ Rx(-0.24503) ├»
    «            │               │           └─────────────┘ └──────────────┘»
    «   q1_9: ───┼───────────────┼──────────────■───────────────■────────────»
    «            │ZZ(46.01)      │              │ZZ(46.01)      │            »
    «  q1_10: ───■───────────────┼──────────────■───────────────┼────────────»
    «                            │ZZ(46.01)                     │ZZ(46.01)   »
    «  q1_11: ──■────────────────■──────────────────────────────■────────────»
    «           │                                                            »
    «  q1_12: ──┼───────────────────────────────■───────────────■────────────»
    «           │                               │ZZ(46.01)      │            »
    «  q1_13: ──┼───────────────■───────────────■───────────────┼────────────»
    «           │               │                               │ZZ(46.01)   »
    «  q1_14: ──┼───────────────┼──────────────■────────────────■────────────»
    «           │ZZ(1.1361)     │              │                             »
    «  q1_15: ──■───────────────┼──────────────┼───────────────■─────────────»
    «                           │              │               │             »
    «  q1_16: ──────────────────┼──────────────┼───────────────┼─────────────»
    «                           │ZZ(1.1361)    │               │             »
    «  q1_17: ──────────────────■──────────────┼───────────────┼─────────────»
    «                                          │ZZ(1.1361)     │             »
    «  q1_18: ─────────────────────────────────■───────────────┼─────────────»
    «                                                          │ZZ(1.1361)   »
    «  q1_19: ─────────────────────────────────────────────────■─────────────»
    «                                                                        »
    «meas: 20/═══════════════════════════════════════════════════════════════»
    «                                                                        »
    «                                                                       »
    «   q1_0: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_1: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_2: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_3: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_4: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_5: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_6: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_7: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «   q1_8: ──────────────────────────────────────────────────────────────»
    «         ┌─────────────┐┌──────────────┐                               »
    «   q1_9: ┤ Rz(-95.428) ├┤ Rx(-0.24503) ├───────────────────────────────»
    «         └─────────────┘├─────────────┬┘┌──────────────┐               »
    «  q1_10: ───■───────────┤ Rz(-95.428) ├─┤ Rx(-0.24503) ├───────────────»
    «            │ZZ(46.01)  ├─────────────┤ ├──────────────┤               »
    «  q1_11: ───■───────────┤ Rz(-95.428) ├─┤ Rx(-0.24503) ├───────────────»
    «                        ├─────────────┤ ├──────────────┤               »
    «  q1_12: ───■───────────┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├───────────────»
    «            │           └─────────────┘ └──────────────┘┌─────────────┐»
    «  q1_13: ───┼──────────────■───────────────■────────────┤ Rz(-94.292) ├»
    «            │              │ZZ(46.01)      │            └─────────────┘»
    «  q1_14: ───┼──────────────■───────────────┼───────────────■───────────»
    «            │ZZ(46.01)                     │ZZ(46.01)      │ZZ(46.01)  »
    «  q1_15: ───■──────────────────────────────■───────────────■───────────»
    «                                                        ┌─────────────┐»
    «  q1_16: ───■──────────────■───────────────■────────────┤ Rz(-93.156) ├»
    «            │ZZ(46.01)     │               │            └─────────────┘»
    «  q1_17: ───■──────────────┼───────────────┼───────────────■───────────»
    «                           │ZZ(46.01)      │               │ZZ(46.01)  »
    «  q1_18: ──────────────────■───────────────┼───────────────■───────────»
    «                                           │ZZ(46.01)                  »
    «  q1_19: ──────────────────────────────────■───────────────────────────»
    «                                                                       »
    «meas: 20/══════════════════════════════════════════════════════════════»
    «                                                                       »
    «                                                                          ░ »
    «   q1_0: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_1: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_2: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_3: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_4: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_5: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_6: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_7: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_8: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «   q1_9: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «  q1_10: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «  q1_11: ─────────────────────────────────────────────────────────────────░─»
    «                                                                          ░ »
    «  q1_12: ─────────────────────────────────────────────────────────────────░─»
    «         ┌──────────────┐                                                 ░ »
    «  q1_13: ┤ Rx(-0.24503) ├─────────────────────────────────────────────────░─»
    «         ├─────────────┬┘┌──────────────┐                                 ░ »
    «  q1_14: ┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├─────────────────────────────────░─»
    «         ├─────────────┤ ├──────────────┤                                 ░ »
    «  q1_15: ┤ Rz(-94.292) ├─┤ Rx(-0.24503) ├─────────────────────────────────░─»
    «         ├─────────────┴┐└──────────────┘                                 ░ »
    «  q1_16: ┤ Rx(-0.24503) ├─────────────────────────────────────────────────░─»
    «         └──────────────┘┌─────────────┐ ┌──────────────┐                 ░ »
    «  q1_17: ───■────────────┤ Rz(-93.156) ├─┤ Rx(-0.24503) ├─────────────────░─»
    «            │            └─────────────┘ ├─────────────┬┘┌──────────────┐ ░ »
    «  q1_18: ───┼───────────────■────────────┤ Rz(-93.156) ├─┤ Rx(-0.24503) ├─░─»
    «            │ZZ(46.01)      │ZZ(46.01)   ├─────────────┤ ├──────────────┤ ░ »
    «  q1_19: ───■───────────────■────────────┤ Rz(-93.156) ├─┤ Rx(-0.24503) ├─░─»
    «                                         └─────────────┘ └──────────────┘ ░ »
    «meas: 20/═══════════════════════════════════════════════════════════════════»
    «                                                                            »
    «         ┌─┐                                                         
    «   q1_0: ┤M├─────────────────────────────────────────────────────────
    «         └╥┘┌─┐                                                      
    «   q1_1: ─╫─┤M├──────────────────────────────────────────────────────
    «          ║ └╥┘┌─┐                                                   
    «   q1_2: ─╫──╫─┤M├───────────────────────────────────────────────────
    «          ║  ║ └╥┘┌─┐                                                
    «   q1_3: ─╫──╫──╫─┤M├────────────────────────────────────────────────
    «          ║  ║  ║ └╥┘┌─┐                                             
    «   q1_4: ─╫──╫──╫──╫─┤M├─────────────────────────────────────────────
    «          ║  ║  ║  ║ └╥┘┌─┐                                          
    «   q1_5: ─╫──╫──╫──╫──╫─┤M├──────────────────────────────────────────
    «          ║  ║  ║  ║  ║ └╥┘┌─┐                                       
    «   q1_6: ─╫──╫──╫──╫──╫──╫─┤M├───────────────────────────────────────
    «          ║  ║  ║  ║  ║  ║ └╥┘┌─┐                                    
    «   q1_7: ─╫──╫──╫──╫──╫──╫──╫─┤M├────────────────────────────────────
    «          ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                                 
    «   q1_8: ─╫──╫──╫──╫──╫──╫──╫──╫─┤M├─────────────────────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                              
    «   q1_9: ─╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├──────────────────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                           
    «  q1_10: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├───────────────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                        
    «  q1_11: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├────────────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                     
    «  q1_12: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├─────────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐                  
    «  q1_13: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├──────────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐               
    «  q1_14: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├───────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐            
    «  q1_15: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├────────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐         
    «  q1_16: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├─────────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐      
    «  q1_17: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├──────
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐   
    «  q1_18: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├───
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐
    «  q1_19: ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├
    «          ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘
    «meas: 20/═╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩═
    «          0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
          ┌───┐┌──────────────────────┐
     q_0: ┤ H ├┤0                     ├
          ├───┤│                      │
     q_1: ┤ H ├┤1                     ├
          ├───┤│                      │
     q_2: ┤ H ├┤2                     ├
          ├───┤│                      │
     q_3: ┤ H ├┤3                     ├
          ├───┤│                      │
     q_4: ┤ H ├┤4                     ├
          ├───┤│                      │
     q_5: ┤ H ├┤5                     ├
          ├───┤│                      │
     q_6: ┤ H ├┤6                     ├
          ├───┤│                      │
     q_7: ┤ H ├┤7                     ├
          ├───┤│   Gate_q_11295816960 │
     q_8: ┤ H ├┤8                     ├
          ├───┤│                      │
     q_9: ┤ H ├┤9                     ├
          └───┘│                      │
    q_10: ─────┤10                    ├
               │                      │
    q_11: ─────┤11                    ├
               │                      │
    q_12: ─────┤12                    ├
               │                      │
    q_13: ─────┤13                    ├
               │                      │
    q_14: ─────┤14                    ├
               │                      │
    q_15: ─────┤15                    ├
               └──────────────────────┘


### An example to find triangle in graph


```python
generator.qasm_generate(triangle_finding_code, verbose=True)
```

    problem type: ProblemType.GRAPH data: Graph with 5 nodes and 7 edges
    -------graph problem type:Triangle--------
    <class 'classical_to_quantum.applications.graph.grover_applications.triangle_finding.TriangleFinding'>
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)]
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)]
    ---doing 4 iterations---
    {   'assignment': '01101',
        'circuit_results': [   {   '00000': 0.030627520754929,
                                   '00001': 0.0306275207549293,
                                   '00010': 0.030627520754929,
                                   '00011': 0.0306275207549292,
                                   '00100': 0.030627520754929,
                                   '00101': 0.0306275207549285,
                                   '00110': 0.030627520754929,
                                   '00111': 0.0372672993689719,
                                   '01000': 0.030627520754929,
                                   '01001': 0.0306275207549293,
                                   '01010': 0.0306275207549291,
                                   '01011': 0.0306275207549292,
                                   '01100': 0.030627520754929,
                                   '01101': 0.037267299368972,
                                   '01110': 0.030627520754929,
                                   '01111': 0.0306275207549285,
                                   '10000': 0.030627520754929,
                                   '10001': 0.0306275207549293,
                                   '10010': 0.030627520754929,
                                   '10011': 0.0306275207549292,
                                   '10100': 0.030627520754929,
                                   '10101': 0.0306275207549285,
                                   '10110': 0.030627520754929,
                                   '10111': 0.0306275207549284,
                                   '11000': 0.030627520754929,
                                   '11001': 0.0306275207549292,
                                   '11010': 0.030627520754929,
                                   '11011': 0.0306275207549292,
                                   '11100': 0.030627520754929,
                                   '11101': 0.0306275207549286,
                                   '11110': 0.030627520754929,
                                   '11111': 0.037267299368972}],
        'iterations': [4],
        'max_probability': 0.037267299368972,
        'oracle_evaluation': True,
        'top_measurement': '01101'}



    
![png](presentation_files/presentation_26_1.png)
    



    
![png](presentation_files/presentation_26_2.png)
    





    {'grover': 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }\ngate mcu1(param0) q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 { cu1(pi/512) q9,q10; cx q9,q8; cu1(-pi/512) q8,q10; cx q9,q8; cu1(pi/512) q8,q10; cx q8,q7; cu1(-pi/512) q7,q10; cx q9,q7; cu1(pi/512) q7,q10; cx q8,q7; cu1(-pi/512) q7,q10; cx q9,q7; cu1(pi/512) q7,q10; cx q7,q6; cu1(-pi/512) q6,q10; cx q9,q6; cu1(pi/512) q6,q10; cx q8,q6; cu1(-pi/512) q6,q10; cx q9,q6; cu1(pi/512) q6,q10; cx q7,q6; cu1(-pi/512) q6,q10; cx q9,q6; cu1(pi/512) q6,q10; cx q8,q6; cu1(-pi/512) q6,q10; cx q9,q6; cu1(pi/512) q6,q10; cx q6,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q8,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q7,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q8,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q6,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q8,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q7,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q8,q5; cu1(-pi/512) q5,q10; cx q9,q5; cu1(pi/512) q5,q10; cx q5,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q7,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q6,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q7,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q5,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q7,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q6,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q7,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q8,q4; cu1(-pi/512) q4,q10; cx q9,q4; cu1(pi/512) q4,q10; cx q4,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q6,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q5,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q6,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q4,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q6,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q5,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q6,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q7,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q8,q3; cu1(-pi/512) q3,q10; cx q9,q3; cu1(pi/512) q3,q10; cx q3,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q5,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q4,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q5,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q3,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q5,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q4,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q5,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q6,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q7,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q8,q2; cu1(-pi/512) q2,q10; cx q9,q2; cu1(pi/512) q2,q10; cx q2,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q4,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q3,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q4,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q2,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q4,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q3,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q4,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q5,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q6,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q7,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q8,q1; cu1(-pi/512) q1,q10; cx q9,q1; cu1(pi/512) q1,q10; cx q1,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q3,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q2,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q3,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q1,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q3,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q2,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q3,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q4,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q5,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q6,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q7,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; cx q8,q0; cu1(-pi/512) q0,q10; cx q9,q0; cu1(pi/512) q0,q10; }\ngate mcx_gray q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 { h q10; mcu1(pi) q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10; h q10; }\ngate gate_Q q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 { x q10; h q10; barrier q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10; ccx q0,q1,q5; mcx q0,q2,q5,q6; ccx q0,q2,q5; mcx q0,q3,q5,q6; ccx q0,q3,q5; mcx q1,q2,q5,q6; ccx q1,q2,q5; mcx q1,q4,q5,q6; ccx q1,q4,q5; mcx q2,q3,q5,q6; ccx q2,q3,q5; mcx q3,q4,q5,q6; ccx q3,q4,q5; ccx q5,q6,q10; ccx q3,q4,q5; mcx q3,q4,q5,q6; ccx q2,q3,q5; mcx q2,q3,q5,q6; ccx q1,q4,q5; mcx q1,q4,q5,q6; ccx q1,q2,q5; mcx q1,q2,q5,q6; ccx q0,q3,q5; mcx q0,q3,q5,q6; ccx q0,q2,q5; mcx q0,q2,q5,q6; ccx q0,q1,q5; barrier q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10; h q10; x q10; h q4; h q3; h q2; h q1; h q0; x q0; x q1; x q2; x q3; x q4; x q5; x q6; x q7; x q8; x q9; x q10; h q10; mcx_gray q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10; h q10; x q0; x q1; x q2; x q3; x q4; x q5; x q6; x q7; x q8; x q9; x q10; h q0; h q1; h q2; h q3; h q4; }\ngate gate_Q_11319179296 q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 { gate_Q q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10; }\nqreg q[11];\nh q[0];\nh q[1];\nh q[2];\nh q[3];\nh q[4];\ngate_Q_11319179296 q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];'}



# Vehicle routing problem
- a case with 5 codes and 2 vehicles (represented by two colors)


```python
vrp_qasm = generator.qasm_generate(vrp_code, verbose=True)
print(qasm2.loads(vrp_qasm.get('qaoa'), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
```

    problem type: ProblemType.GRAPH data: Graph with 5 nodes and 10 edges
    -------graph problem type:VRP--------
    <class 'classical_to_quantum.applications.graph.Ising.Ising'>
    [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    {'angles': [0.578363766448, 0.348846960162, 0.119190749265, 0.12003484033, 0.359742123326, 0.566509948156], 'cost': 166.71, 'measurement_outcomes': {'1000010001': 1, '0110011001': 1, '1000010000': 1, '1110111001': 1, '0011111001': 1, '0000010110': 1, '1001011010': 1, '1110111011': 1, '0000001000': 1, '1011010011': 1, '1001001010': 1, '1101101110': 1, '0100111011': 1, '0110010111': 1, '0001101111': 1, '1010111000': 1, '1011011111': 1, '1001110101': 2, '1010010000': 1, '1010000000': 1, '1111001000': 1, '1001010111': 1, '1100011000': 1, '1001000000': 2, '0100111100': 1, '1100000000': 1, '1110010100': 1, '0000011000': 1, '1001101010': 1, '1100100010': 1, '1100011010': 1, '1011000110': 1, '1000110011': 1, '1111110000': 1, '1101111111': 1, '1100011110': 1, '0100011000': 1, '0110001010': 1, '0001011100': 1, '0100001000': 1, '0000110111': 1, '1110001011': 1, '0100100010': 1, '1000010110': 4, '1000100100': 1, '1000111010': 3, '1101001000': 1, '1000111110': 1, '1100001100': 2, '1000000010': 1, '0001110011': 1, '1001100011': 1, '0011110010': 1, '1101101010': 1, '1100000001': 1, '1001110111': 1, '0100010011': 1, '0001111000': 1, '0001111010': 1, '0111011001': 1, '1111001110': 1, '1000011000': 1, '0001111011': 1, '1101100011': 1, '1001010011': 1, '1011101110': 1, '0010101110': 1, '0110001011': 2, '1001111011': 2, '1001010100': 1, '1010111011': 1, '0100011010': 1, '0000010100': 2, '1100001010': 3, '1000001000': 4, '1101011111': 1, '1100010001': 1, '1101011011': 1, '0100111000': 2, '0100110001': 1, '1110000000': 1, '1001100111': 1, '1101111000': 1}, 'job_id': 'a6ec6956-d590-432a-9092-7880767eec9e', 'eval_number': 432}



    
![png](presentation_files/presentation_28_1.png)
    


             ┌───┐                                                        »
       q3_0: ┤ H ├─■─────────────■─────────────■──────────────────────────»
             ├───┤ │ZZ(0.48014)  │             │                          »
       q3_1: ┤ H ├─■─────────────┼─────────────┼─────────────■────────────»
             ├───┤               │ZZ(0.48014)  │             │ZZ(0.48014) »
       q3_2: ┤ H ├───────────────■─────────────┼─────────────■────────────»
             ├───┤                             │ZZ(0.48014)               »
       q3_3: ┤ H ├─────────────────────────────■──────────────────────────»
             ├───┤                                                        »
       q3_4: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q3_5: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q3_6: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q3_7: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q3_8: ┤ H ├────────────────────────────────────────────────────────»
             ├───┤                                                        »
       q3_9: ┤ H ├────────────────────────────────────────────────────────»
             └───┘                                                        »
    meas: 10/═════════════════════════════════════════════════════════════»
                                                                          »
    «                                                                 »
    «   q3_0: ───────────────■───────────────────────────■────────────»
    «                        │                           │            »
    «   q3_1: ─■─────────────┼───────────────────────────┼────────────»
    «          │             │                           │            »
    «   q3_2: ─┼─────────────┼─────────────■─────────────┼────────────»
    «          │ZZ(0.48014)  │             │ZZ(0.48014)  │            »
    «   q3_3: ─■─────────────┼─────────────■─────────────┼────────────»
    «                        │ZZ(0.48014)                │            »
    «   q3_4: ───────────────■───────────────────────────┼────────────»
    «                                                    │ZZ(0.48014) »
    «   q3_5: ───────────────────────────────────────────■────────────»
    «                                                                 »
    «   q3_6: ────────────────────────────────────────────────────────»
    «                                                                 »
    «   q3_7: ────────────────────────────────────────────────────────»
    «                                                                 »
    «   q3_8: ────────────────────────────────────────────────────────»
    «                                                                 »
    «   q3_9: ────────────────────────────────────────────────────────»
    «                                                                 »
    «meas: 10/════════════════════════════════════════════════════════»
    «                                                                 »
    «                       ┌─────────────┐┌─────────────┐              »
    «   q3_0: ─■────────────┤ Rz(0.72021) ├┤ Rx(-1.1567) ├──────────────»
    «          │            └─────────────┘└─────────────┘              »
    «   q3_1: ─┼───────────────────────────────────────────■────────────»
    «          │                                           │            »
    «   q3_2: ─┼───────────────────────────────────────────┼────────────»
    «          │                                           │            »
    «   q3_3: ─┼───────────────────────────────────────────┼────────────»
    «          │                                           │ZZ(0.48014) »
    «   q3_4: ─┼──────────────■──────────────■─────────────■────────────»
    «          │              │ZZ(0.48014)   │                          »
    «   q3_5: ─┼──────────────■──────────────┼─────────────■────────────»
    «          │ZZ(0.48014)                  │ZZ(0.48014)  │ZZ(0.48014) »
    «   q3_6: ─■─────────────────────────────■─────────────■────────────»
    «                                                                   »
    «   q3_7: ──────────────────────────────────────────────────────────»
    «                                                                   »
    «   q3_8: ──────────────────────────────────────────────────────────»
    «                                                                   »
    «   q3_9: ──────────────────────────────────────────────────────────»
    «                                                                   »
    «meas: 10/══════════════════════════════════════════════════════════»
    «                                                                   »
    «                                                                 »
    «   q3_0: ────────────────────────────────────────────────────────»
    «                                                                 »
    «   q3_1: ─■─────────────────────────────────────────■────────────»
    «          │                                         │            »
    «   q3_2: ─┼─────────────■───────────────────────────┼────────────»
    «          │             │                           │            »
    «   q3_3: ─┼─────────────┼─────────────■─────────────┼────────────»
    «          │             │             │             │            »
    «   q3_4: ─┼─────────────┼─────────────┼─────────────┼────────────»
    «          │             │ZZ(0.48014)  │             │            »
    «   q3_5: ─┼─────────────■─────────────┼─────────────┼────────────»
    «          │                           │ZZ(0.48014)  │            »
    «   q3_6: ─┼───────────────────────────■─────────────┼────────────»
    «          │ZZ(0.48014)                              │            »
    «   q3_7: ─■─────────────────────────────────────────┼────────────»
    «                                                    │ZZ(0.48014) »
    «   q3_8: ───────────────────────────────────────────■────────────»
    «                                                                 »
    «   q3_9: ────────────────────────────────────────────────────────»
    «                                                                 »
    «meas: 10/════════════════════════════════════════════════════════»
    «                                                                 »
    «                                                                     »
    «   q3_0: ──────────────────────────────────■─────────────────────────»
    «         ┌──────────────┐┌─────────────┐   │ZZ(1.439)                »
    «   q3_1: ┤ Rz(-0.48014) ├┤ Rx(-1.1567) ├───■─────────────────────────»
    «         └──────────────┘└─────────────┘                             »
    «   q3_2: ───────────────────────────────────────────────■────────────»
    «                                                        │            »
    «   q3_3: ───────────────────────────────────────────────┼────────────»
    «                                        ┌─────────────┐ │            »
    «   q3_4: ──■───────────────■────────────┤ Rz(-3.0009) ├─┼────────────»
    «           │               │            └─────────────┘ │            »
    «   q3_5: ──┼───────────────┼────────────────────────────┼────────────»
    «           │               │                            │            »
    «   q3_6: ──┼───────────────┼────────────────────────────┼────────────»
    «           │ZZ(0.48014)    │                            │ZZ(0.48014) »
    «   q3_7: ──■───────────────┼──────────────■─────────────■────────────»
    «                           │ZZ(0.48014)   │ZZ(0.48014)               »
    «   q3_8: ──────────────────■──────────────■──────────────────────────»
    «                                                                     »
    «   q3_9: ────────────────────────────────────────────────────────────»
    «                                                                     »
    «meas: 10/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                                                   »
    «   q3_0: ──────────────────────────────────────────────────────────»
    «                                                                   »
    «   q3_1: ──────────────────────────────────────────────────────────»
    «                                                    ┌─────────────┐»
    «   q3_2: ──────────────────────────────■────────────┤ Rz(-1.6805) ├»
    «                                       │            └─────────────┘»
    «   q3_3: ─■────────────────────────────┼───────────────────────────»
    «          │            ┌─────────────┐ │                           »
    «   q3_4: ─┼────────────┤ Rx(-1.1567) ├─┼───────────────────────────»
    «          │            └─────────────┘ │                           »
    «   q3_5: ─┼──────────────■─────────────┼───────────────────────────»
    «          │              │             │                           »
    «   q3_6: ─┼──────────────┼─────────────┼──────────────■────────────»
    «          │              │ZZ(0.48014)  │              │            »
    «   q3_7: ─┼──────────────■─────────────┼──────────────┼────────────»
    «          │ZZ(0.48014)                 │              │ZZ(0.48014) »
    «   q3_8: ─■────────────────────────────┼──────────────■────────────»
    «                                       │ZZ(0.48014)                »
    «   q3_9: ──────────────────────────────■───────────────────────────»
    «                                                                   »
    «meas: 10/══════════════════════════════════════════════════════════»
    «                                                                   »
    «                                                                    »
    «   q3_0: ──────────────────■────────────────────────────────────────»
    «                           │                                        »
    «   q3_1: ──────────────────┼─────────────■──────────────────────────»
    «         ┌─────────────┐   │ZZ(1.439)    │ZZ(1.439)                 »
    «   q3_2: ┤ Rx(-1.1567) ├───■─────────────■──────────────────────────»
    «         └─────────────┘                             ┌─────────────┐»
    «   q3_3: ───────────────────────────────■────────────┤ Rz(-2.8808) ├»
    «                                        │            └─────────────┘»
    «   q3_4: ───────────────────────────────┼───────────────────────────»
    «                        ┌─────────────┐ │            ┌─────────────┐»
    «   q3_5: ──■────────────┤ Rz(-4.2012) ├─┼────────────┤ Rx(-1.1567) ├»
    «           │            └─────────────┘ │            └─────────────┘»
    «   q3_6: ──┼────────────────────────────┼───────────────────────────»
    «           │                            │            ┌─────────────┐»
    «   q3_7: ──┼──────────────■─────────────┼────────────┤ Rz(-1.8005) ├»
    «           │              │             │            └─────────────┘»
    «   q3_8: ──┼──────────────┼─────────────┼───────────────────────────»
    «           │ZZ(0.48014)   │ZZ(0.48014)  │ZZ(0.48014)                »
    «   q3_9: ──■──────────────■─────────────■───────────────────────────»
    «                                                                    »
    «meas: 10/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                                     »
    «   q3_0: ──────────────────■─────────────────────────────■───────────»
    «                           │                             │           »
    «   q3_1: ──────────────────┼──────────────■──────────────┼───────────»
    «                           │              │              │           »
    «   q3_2: ──────────────────┼──────────────┼──────────────┼───────────»
    «         ┌─────────────┐   │ZZ(1.439)     │ZZ(1.439)     │           »
    «   q3_3: ┤ Rx(-1.1567) ├───■──────────────■──────────────┼───────────»
    «         └─────────────┘                                 │ZZ(1.439)  »
    «   q3_4: ────────────────────────────────────────────────■───────────»
    «                                                                     »
    «   q3_5: ────────────────────────────────────────────────────────────»
    «                        ┌─────────────┐┌─────────────┐               »
    «   q3_6: ──■────────────┤ Rz(-5.4016) ├┤ Rx(-1.1567) ├───────────────»
    «           │            ├─────────────┤└─────────────┘               »
    «   q3_7: ──┼────────────┤ Rx(-1.1567) ├──────────────────────────────»
    «           │            └─────────────┘ ┌────────────┐┌─────────────┐»
    «   q3_8: ──┼──────────────■─────────────┤ Rz(-3.601) ├┤ Rx(-1.1567) ├»
    «           │ZZ(0.48014)   │ZZ(0.48014) ┌┴────────────┤├─────────────┤»
    «   q3_9: ──■──────────────■────────────┤ Rz(-2.4007) ├┤ Rx(-1.1567) ├»
    «                                       └─────────────┘└─────────────┘»
    «meas: 10/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                             ┌────────────┐┌──────────────┐»
    «   q3_0: ─────────────■───────────■──────────┤ Rz(2.1585) ├┤ Rx(-0.69769) ├»
    «                      │           │          └────────────┘└──────────────┘»
    «   q3_1: ─────────────┼───────────┼────────────────────────────────────────»
    «                      │           │                                        »
    «   q3_2: ─■───────────┼───────────┼────────────────────────────────────────»
    «          │ZZ(1.439)  │           │                                        »
    «   q3_3: ─■───────────┼───────────┼────────────────────────────────────────»
    «                      │           │                                        »
    «   q3_4: ─────────────┼───────────┼────────────■──────────────■────────────»
    «                      │ZZ(1.439)  │            │ZZ(1.439)     │            »
    «   q3_5: ─────────────■───────────┼────────────■──────────────┼────────────»
    «                                  │ZZ(1.439)                  │ZZ(1.439)   »
    «   q3_6: ─────────────────────────■───────────────────────────■────────────»
    «                                                                           »
    «   q3_7: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «   q3_8: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «   q3_9: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «meas: 10/══════════════════════════════════════════════════════════════════»
    «                                                                           »
    «                                                                     »
    «   q3_0: ────────────────────────────────────────────────────────────»
    «                                                                     »
    «   q3_1: ─■───────────■───────────────────────────────────■──────────»
    «          │           │                                   │          »
    «   q3_2: ─┼───────────┼───────────■───────────────────────┼──────────»
    «          │           │           │                       │          »
    «   q3_3: ─┼───────────┼───────────┼───────────■───────────┼──────────»
    «          │ZZ(1.439)  │           │           │           │          »
    «   q3_4: ─■───────────┼───────────┼───────────┼───────────┼──────────»
    «                      │           │ZZ(1.439)  │           │          »
    «   q3_5: ─■───────────┼───────────■───────────┼───────────┼──────────»
    «          │ZZ(1.439)  │                       │ZZ(1.439)  │          »
    «   q3_6: ─■───────────┼───────────────────────■───────────┼──────────»
    «                      │ZZ(1.439)                          │          »
    «   q3_7: ─────────────■───────────────────────────────────┼──────────»
    «                                                          │ZZ(1.439) »
    «   q3_8: ─────────────────────────────────────────────────■──────────»
    «                                                                     »
    «   q3_9: ────────────────────────────────────────────────────────────»
    «                                                                     »
    «meas: 10/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                                                              »
    «   q3_0: ─────────────────────────────────■───────────────────────────────────»
    «         ┌────────────┐┌──────────────┐   │ZZ(2.266)                          »
    «   q3_1: ┤ Rz(-1.439) ├┤ Rx(-0.69769) ├───■───────────────────────────────────»
    «         └────────────┘└──────────────┘                                       »
    «   q3_2: ──────────────────────────────────────────────■──────────────────────»
    «                                                       │                      »
    «   q3_3: ──────────────────────────────────────────────┼───────────■──────────»
    «                                       ┌─────────────┐ │           │          »
    «   q3_4: ──■──────────────■────────────┤ Rz(-8.9936) ├─┼───────────┼──────────»
    «           │              │            └─────────────┘ │           │          »
    «   q3_5: ──┼──────────────┼────────────────────────────┼───────────┼──────────»
    «           │              │                            │           │          »
    «   q3_6: ──┼──────────────┼────────────────────────────┼───────────┼──────────»
    «           │ZZ(1.439)     │                            │ZZ(1.439)  │          »
    «   q3_7: ──■──────────────┼───────────────■────────────■───────────┼──────────»
    «                          │ZZ(1.439)      │ZZ(1.439)               │ZZ(1.439) »
    «   q3_8: ─────────────────■───────────────■────────────────────────■──────────»
    «                                                                              »
    «   q3_9: ─────────────────────────────────────────────────────────────────────»
    «                                                                              »
    «meas: 10/═════════════════════════════════════════════════════════════════════»
    «                                                                              »
    «                                                                    »
    «   q3_0: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «   q3_1: ───────────────────────────────────────────────────────────»
    «                                     ┌─────────────┐┌──────────────┐»
    «   q3_2: ─────────────────■──────────┤ Rz(-5.0364) ├┤ Rx(-0.69769) ├»
    «                          │          └─────────────┘└──────────────┘»
    «   q3_3: ─────────────────┼─────────────────────────────────────────»
    «         ┌──────────────┐ │                                         »
    «   q3_4: ┤ Rx(-0.69769) ├─┼─────────────────────────────────────────»
    «         └──────────────┘ │                                         »
    «   q3_5: ───■─────────────┼────────────────────────────■────────────»
    «            │             │                            │            »
    «   q3_6: ───┼─────────────┼─────────────■──────────────┼────────────»
    «            │ZZ(1.439)    │             │              │            »
    «   q3_7: ───■─────────────┼─────────────┼──────────────┼────────────»
    «                          │             │ZZ(1.439)     │            »
    «   q3_8: ─────────────────┼─────────────■──────────────┼────────────»
    «                          │ZZ(1.439)                   │ZZ(1.439)   »
    «   q3_9: ─────────────────■────────────────────────────■────────────»
    «                                                                    »
    «meas: 10/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                                    »
    «   q3_0: ───■───────────────────────────────────────────────────────»
    «            │                                                       »
    «   q3_1: ───┼────────────■──────────────────────────────────────────»
    «            │ZZ(2.266)   │ZZ(2.266)                                 »
    «   q3_2: ───■────────────■──────────────────────────────────────────»
    «                                    ┌─────────────┐ ┌──────────────┐»
    «   q3_3: ────────────────■──────────┤ Rz(-8.6338) ├─┤ Rx(-0.69769) ├»
    «                         │          └─────────────┘ └──────────────┘»
    «   q3_4: ────────────────┼──────────────────────────────────────────»
    «         ┌─────────────┐ │          ┌──────────────┐                »
    «   q3_5: ┤ Rz(-12.591) ├─┼──────────┤ Rx(-0.69769) ├────────────────»
    «         └─────────────┘ │          └──────────────┘                »
    «   q3_6: ────────────────┼─────────────────────────────■────────────»
    «                         │          ┌─────────────┐    │            »
    «   q3_7: ───■────────────┼──────────┤ Rz(-5.3961) ├────┼────────────»
    «            │            │          └─────────────┘    │            »
    «   q3_8: ───┼────────────┼─────────────────────────────┼────────────»
    «            │ZZ(1.439)   │ZZ(1.439)                    │ZZ(1.439)   »
    «   q3_9: ───■────────────■─────────────────────────────■────────────»
    «                                                                    »
    «meas: 10/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                                     »
    «   q3_0: ───■───────────────────────────────■────────────────────────»
    «            │                               │                        »
    «   q3_1: ───┼───────────────■───────────────┼────────────────────────»
    «            │               │               │                        »
    «   q3_2: ───┼───────────────┼───────────────┼─────────────■──────────»
    «            │ZZ(2.266)      │ZZ(2.266)      │             │ZZ(2.266) »
    «   q3_3: ───■───────────────■───────────────┼─────────────■──────────»
    «                                            │ZZ(2.266)               »
    «   q3_4: ───────────────────────────────────■────────────────────────»
    «                                                                     »
    «   q3_5: ────────────────────────────────────────────────────────────»
    «         ┌─────────────┐ ┌──────────────┐                            »
    «   q3_6: ┤ Rz(-16.188) ├─┤ Rx(-0.69769) ├────────────────────────────»
    «         ├─────────────┴┐└──────────────┘                            »
    «   q3_7: ┤ Rx(-0.69769) ├────────────────────────────────────────────»
    «         └──────────────┘┌─────────────┐ ┌──────────────┐            »
    «   q3_8: ───■────────────┤ Rz(-10.792) ├─┤ Rx(-0.69769) ├────────────»
    «            │ZZ(1.439)   ├─────────────┤ ├──────────────┤            »
    «   q3_9: ───■────────────┤ Rz(-7.1948) ├─┤ Rx(-0.69769) ├────────────»
    «                         └─────────────┘ └──────────────┘            »
    «meas: 10/════════════════════════════════════════════════════════════»
    «                                                                     »
    «                                 ┌────────────┐┌──────────────┐            »
    «   q3_0: ─■───────────■──────────┤ Rz(3.3991) ├┤ Rx(-0.23838) ├────────────»
    «          │           │          └────────────┘└──────────────┘            »
    «   q3_1: ─┼───────────┼─────────────────────────────────────────■──────────»
    «          │           │                                         │          »
    «   q3_2: ─┼───────────┼─────────────────────────────────────────┼──────────»
    «          │           │                                         │          »
    «   q3_3: ─┼───────────┼─────────────────────────────────────────┼──────────»
    «          │           │                                         │ZZ(2.266) »
    «   q3_4: ─┼───────────┼────────────■──────────────■─────────────■──────────»
    «          │ZZ(2.266)  │            │ZZ(2.266)     │                        »
    «   q3_5: ─■───────────┼────────────■──────────────┼─────────────■──────────»
    «                      │ZZ(2.266)                  │ZZ(2.266)    │ZZ(2.266) »
    «   q3_6: ─────────────■───────────────────────────■─────────────■──────────»
    «                                                                           »
    «   q3_7: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «   q3_8: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «   q3_9: ──────────────────────────────────────────────────────────────────»
    «                                                                           »
    «meas: 10/══════════════════════════════════════════════════════════════════»
    «                                                                           »
    «                                                                       »
    «   q3_0: ──────────────────────────────────────────────────────────────»
    «                                                         ┌────────────┐»
    «   q3_1: ─■───────────────────────────────────■──────────┤ Rz(-2.266) ├»
    «          │                                   │          └────────────┘»
    «   q3_2: ─┼───────────■───────────────────────┼────────────────────────»
    «          │           │                       │                        »
    «   q3_3: ─┼───────────┼───────────■───────────┼────────────────────────»
    «          │           │           │           │                        »
    «   q3_4: ─┼───────────┼───────────┼───────────┼────────────■───────────»
    «          │           │ZZ(2.266)  │           │            │           »
    «   q3_5: ─┼───────────■───────────┼───────────┼────────────┼───────────»
    «          │                       │ZZ(2.266)  │            │           »
    «   q3_6: ─┼───────────────────────■───────────┼────────────┼───────────»
    «          │ZZ(2.266)                          │            │ZZ(2.266)  »
    «   q3_7: ─■───────────────────────────────────┼────────────■───────────»
    «                                              │ZZ(2.266)               »
    «   q3_8: ─────────────────────────────────────■────────────────────────»
    «                                                                       »
    «   q3_9: ──────────────────────────────────────────────────────────────»
    «                                                                       »
    «meas: 10/══════════════════════════════════════════════════════════════»
    «                                                                       »
    «                                                                »
    «   q3_0: ───────────────────────────────────────────────────────»
    «         ┌──────────────┐                                       »
    «   q3_1: ┤ Rx(-0.23838) ├───────────────────────────────────────»
    «         └──────────────┘                                       »
    «   q3_2: ────────────────────────────────■──────────────────────»
    «                                         │                      »
    «   q3_3: ────────────────────────────────┼───────────■──────────»
    «                         ┌─────────────┐ │           │          »
    «   q3_4: ───■────────────┤ Rz(-14.163) ├─┼───────────┼──────────»
    «            │            └─────────────┘ │           │          »
    «   q3_5: ───┼────────────────────────────┼───────────┼──────────»
    «            │                            │           │          »
    «   q3_6: ───┼────────────────────────────┼───────────┼──────────»
    «            │                            │ZZ(2.266)  │          »
    «   q3_7: ───┼───────────────■────────────■───────────┼──────────»
    «            │ZZ(2.266)      │ZZ(2.266)               │ZZ(2.266) »
    «   q3_8: ───■───────────────■────────────────────────■──────────»
    «                                                                »
    «   q3_9: ───────────────────────────────────────────────────────»
    «                                                                »
    «meas: 10/═══════════════════════════════════════════════════════»
    «                                                                »
    «                                                                    »
    «   q3_0: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «   q3_1: ───────────────────────────────────────────────────────────»
    «                                     ┌─────────────┐┌──────────────┐»
    «   q3_2: ─────────────────■──────────┤ Rz(-7.9311) ├┤ Rx(-0.23838) ├»
    «                          │          └─────────────┘└──────────────┘»
    «   q3_3: ─────────────────┼─────────────────────────────────────────»
    «         ┌──────────────┐ │                                         »
    «   q3_4: ┤ Rx(-0.23838) ├─┼─────────────────────────────────────────»
    «         └──────────────┘ │                                         »
    «   q3_5: ───■─────────────┼────────────────────────────■────────────»
    «            │             │                            │            »
    «   q3_6: ───┼─────────────┼─────────────■──────────────┼────────────»
    «            │ZZ(2.266)    │             │              │            »
    «   q3_7: ───■─────────────┼─────────────┼──────────────┼────────────»
    «                          │             │ZZ(2.266)     │            »
    «   q3_8: ─────────────────┼─────────────■──────────────┼────────────»
    «                          │ZZ(2.266)                   │ZZ(2.266)   »
    «   q3_9: ─────────────────■────────────────────────────■────────────»
    «                                                                    »
    «meas: 10/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                                    »
    «   q3_0: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «   q3_1: ───────────────────────────────────────────────────────────»
    «                                                                    »
    «   q3_2: ───────────────────────────────────────────────────────────»
    «                                    ┌─────────────┐ ┌──────────────┐»
    «   q3_3: ────────────────■──────────┤ Rz(-13.596) ├─┤ Rx(-0.23838) ├»
    «                         │          └─────────────┘ └──────────────┘»
    «   q3_4: ────────────────┼──────────────────────────────────────────»
    «         ┌─────────────┐ │          ┌──────────────┐                »
    «   q3_5: ┤ Rz(-19.828) ├─┼──────────┤ Rx(-0.23838) ├────────────────»
    «         └─────────────┘ │          └──────────────┘                »
    «   q3_6: ────────────────┼─────────────────────────────■────────────»
    «                         │          ┌─────────────┐    │            »
    «   q3_7: ───■────────────┼──────────┤ Rz(-8.4976) ├────┼────────────»
    «            │            │          └─────────────┘    │            »
    «   q3_8: ───┼────────────┼─────────────────────────────┼────────────»
    «            │ZZ(2.266)   │ZZ(2.266)                    │ZZ(2.266)   »
    «   q3_9: ───■────────────■─────────────────────────────■────────────»
    «                                                                    »
    «meas: 10/═══════════════════════════════════════════════════════════»
    «                                                                    »
    «                                                          ░ ┌─┐               »
    «   q3_0: ─────────────────────────────────────────────────░─┤M├───────────────»
    «                                                          ░ └╥┘┌─┐            »
    «   q3_1: ─────────────────────────────────────────────────░──╫─┤M├────────────»
    «                                                          ░  ║ └╥┘┌─┐         »
    «   q3_2: ─────────────────────────────────────────────────░──╫──╫─┤M├─────────»
    «                                                          ░  ║  ║ └╥┘┌─┐      »
    «   q3_3: ─────────────────────────────────────────────────░──╫──╫──╫─┤M├──────»
    «                                                          ░  ║  ║  ║ └╥┘┌─┐   »
    «   q3_4: ─────────────────────────────────────────────────░──╫──╫──╫──╫─┤M├───»
    «                                                          ░  ║  ║  ║  ║ └╥┘┌─┐»
    «   q3_5: ─────────────────────────────────────────────────░──╫──╫──╫──╫──╫─┤M├»
    «         ┌─────────────┐ ┌──────────────┐                 ░  ║  ║  ║  ║  ║ └╥┘»
    «   q3_6: ┤ Rz(-25.493) ├─┤ Rx(-0.23838) ├─────────────────░──╫──╫──╫──╫──╫──╫─»
    «         ├─────────────┴┐└──────────────┘                 ░  ║  ║  ║  ║  ║  ║ »
    «   q3_7: ┤ Rx(-0.23838) ├─────────────────────────────────░──╫──╫──╫──╫──╫──╫─»
    «         └──────────────┘┌─────────────┐ ┌──────────────┐ ░  ║  ║  ║  ║  ║  ║ »
    «   q3_8: ───■────────────┤ Rz(-16.995) ├─┤ Rx(-0.23838) ├─░──╫──╫──╫──╫──╫──╫─»
    «            │ZZ(2.266)   └┬────────────┤ ├──────────────┤ ░  ║  ║  ║  ║  ║  ║ »
    «   q3_9: ───■─────────────┤ Rz(-11.33) ├─┤ Rx(-0.23838) ├─░──╫──╫──╫──╫──╫──╫─»
    «                          └────────────┘ └──────────────┘ ░  ║  ║  ║  ║  ║  ║ »
    «meas: 10/════════════════════════════════════════════════════╩══╩══╩══╩══╩══╩═»
    «                                                             0  1  2  3  4  5 »
    «                     
    «   q3_0: ────────────
    «                     
    «   q3_1: ────────────
    «                     
    «   q3_2: ────────────
    «                     
    «   q3_3: ────────────
    «                     
    «   q3_4: ────────────
    «                     
    «   q3_5: ────────────
    «         ┌─┐         
    «   q3_6: ┤M├─────────
    «         └╥┘┌─┐      
    «   q3_7: ─╫─┤M├──────
    «          ║ └╥┘┌─┐   
    «   q3_8: ─╫──╫─┤M├───
    «          ║  ║ └╥┘┌─┐
    «   q3_9: ─╫──╫──╫─┤M├
    «          ║  ║  ║ └╥┘
    «meas: 10/═╩══╩══╩══╩═
    «          6  7  8  9 


## Quantum Multiplication 


```python
multiplication_qasm = generator.qasm_generate(multiplication_code, verbose=True)
print(qasm2.loads(multiplication_qasm.get('QFT'), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
```

    problem type: ProblemType.ARITHMETICS data: {'left': 4, 'right': 5}
    quantum Multiplication result: 20
               ┌────────────────────────┐                  
     q_0: ─────┤0                       ├──────────────────
               │                        │                  
     q_1: ─────┤1                       ├──────────────────
          ┌───┐│                        │                  
     q_2: ┤ X ├┤2                       ├──────────────────
          ├───┤│                        │                  
     q_3: ┤ X ├┤3                       ├──────────────────
          └───┘│                        │                  
     q_4: ─────┤4                       ├──────────────────
          ┌───┐│                        │                  
     q_5: ┤ X ├┤5                       ├──────────────────
          └───┘│   Gate_rgqftmultiplier │┌─┐               
     q_6: ─────┤6                       ├┤M├───────────────
               │                        │└╥┘┌─┐            
     q_7: ─────┤7                       ├─╫─┤M├────────────
               │                        │ ║ └╥┘┌─┐         
     q_8: ─────┤8                       ├─╫──╫─┤M├─────────
               │                        │ ║  ║ └╥┘┌─┐      
     q_9: ─────┤9                       ├─╫──╫──╫─┤M├──────
               │                        │ ║  ║  ║ └╥┘┌─┐   
    q_10: ─────┤10                      ├─╫──╫──╫──╫─┤M├───
               │                        │ ║  ║  ║  ║ └╥┘┌─┐
    q_11: ─────┤11                      ├─╫──╫──╫──╫──╫─┤M├
               └────────────────────────┘ ║  ║  ║  ║  ║ └╥┘
     c: 6/════════════════════════════════╩══╩══╩══╩══╩══╩═
                                          0  1  2  3  4  5 


# Quantum conjunctive normal formula


```python
cnf_qasm = generator.qasm_generate(cnf_code, verbose=True)
print(qasm2.loads(cnf_qasm.get('grover'), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
```

    problem type: ProblemType.CNF data: [[-1, -2, -3], [1, -2, 3], [1, 2, -3], [1, -2, -3], [-1, 2, 3]]
    {   'assignment': '000',
        'circuit_results': [   {   '000': 0.2812499999999996,
                                   '001': 0.0312499999999999,
                                   '010': 0.0312499999999999,
                                   '011': 0.2812499999999995,
                                   '100': 0.03125,
                                   '101': 0.2812499999999995,
                                   '110': 0.0312499999999999,
                                   '111': 0.03125}],
        'iterations': [1],
        'max_probability': 0.2812499999999996,
        'oracle_evaluation': True,
        'top_measurement': '000'}
         ┌───┐┌────────────────────┐┌─────────────────────┐┌─┐      
    q_0: ┤ H ├┤0                   ├┤0                    ├┤M├──────
         ├───┤│                    ││                     │└╥┘┌─┐   
    q_1: ┤ H ├┤1 Gate_q_6248185088 ├┤1 Gate_q_11529635968 ├─╫─┤M├───
         ├───┤│                    ││                     │ ║ └╥┘┌─┐
    q_2: ┤ H ├┤2                   ├┤2                    ├─╫──╫─┤M├
         └───┘└────────────────────┘└─────────────────────┘ ║  ║ └╥┘
    c: 3/═══════════════════════════════════════════════════╩══╩══╩═
                                                            0  1  2 


## Quantum factorization (Grover and Shor algorithms)


```python
factor_qasm = generator.qasm_generate(factor_code, verbose=True)
factor_grover_qasm = factor_qasm.get('grover')
factor_circuit = qasm2.loads(factor_grover_qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
print(factor_circuit)
```

    problem type: ProblemType.FACTOR data: {'composite number': 35}
    [{'101': 0.1689453124999865}, {'111': 0.1689453124999864}]
          ┌───┐┌──────────────────────┐┌─┐      
     q_0: ┤ H ├┤0                     ├┤M├──────
          ├───┤│                      │└╥┘┌─┐   
     q_1: ┤ H ├┤1                     ├─╫─┤M├───
          ├───┤│                      │ ║ └╥┘┌─┐
     q_2: ┤ H ├┤2                     ├─╫──╫─┤M├
          ├───┤│                      │ ║  ║ └╥┘
     q_3: ┤ H ├┤3                     ├─╫──╫──╫─
          ├───┤│                      │ ║  ║  ║ 
     q_4: ┤ H ├┤4                     ├─╫──╫──╫─
          ├───┤│                      │ ║  ║  ║ 
     q_5: ┤ H ├┤5                     ├─╫──╫──╫─
          └───┘│                      │ ║  ║  ║ 
     q_6: ─────┤6  Gate_q_11275946400 ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
     q_7: ─────┤7                     ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
     q_8: ─────┤8                     ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
     q_9: ─────┤9                     ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
    q_10: ─────┤10                    ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
    q_11: ─────┤11                    ├─╫──╫──╫─
               │                      │ ║  ║  ║ 
    q_12: ─────┤12                    ├─╫──╫──╫─
               └──────────────────────┘ ║  ║  ║ 
     c: 3/══════════════════════════════╩══╩══╩═
                                        0  1  2 


### Retrieve generated qasm code, run on local simulators and interpret results...
- 5 and 7 with probability of $19.5\%$, which is correct answer $5*7=35$


```python
res = generator.run_qasm_simulator(factor_qasm.get('grover'))
plot_histogram(res.quasi_dists)
```




    
![png](presentation_files/presentation_36_0.png)
    




```python

```

