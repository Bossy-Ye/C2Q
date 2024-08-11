# Classi|Q> Project!!!
Trying to bridge the gap between classic and quantum, for those who are not familiar with quantum computing.
This project is both practical and educational. 

## License
This project takes Qiskit open source project as a reference, thanks to community.
A copy of this license is included in the `LICENSE` file.

## Attribution
- Qiskit: https://qiskit.org/
- License: http://www.apache.org/licenses/LICENSE-2.0
- Author: Boshuai Ye, email: boshuaiye@gmail.com
### Workflow of Classi|Q>
![alt text](./assets/workflow.png "Title")

### How we translate classical problem into quantum???
We aim to analyze the given classical code by extracting its Abstract Syntax Tree (AST), traversing it to identify the type of problem being solved, and then capturing the original data. 
The next step is to convert this input data into a format suitable for 
quantum computation. Currently, we are focusing on converting NP problems 
to CNF (Conjunctive Normal Form) and utilizing the Quantum Approximate 
Optimization Algorithm (QAOA). For these cases, oracles tailored to 
different types of problems will be required. And also a translator that gives readable output.

### Limitedness
Inherent complexity to translate original problem into CNF...




