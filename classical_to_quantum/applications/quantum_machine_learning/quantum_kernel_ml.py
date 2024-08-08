from qiskit._accelerate import qasm3
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, state_fidelity
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from qiskit import qasm3, qasm2


def calculate_kernel(feature_map, x_data, y_data=None):
    if y_data is None:
        y_data = x_data

    x_circuits = [feature_map.assign_parameters(x) for x in x_data]
    y_circuits = [feature_map.assign_parameters(y) for y in y_data]

    kernel = np.zeros((y_data.shape[0], x_data.shape[0]))

    for i, x_c in enumerate(x_circuits):
        for j, y_c in enumerate(y_circuits):
            sv_x = Statevector.from_instruction(x_c)
            sv_y = Statevector.from_instruction(y_c)
            fidelity = state_fidelity(sv_x, sv_y)

            kernel[j, i] = np.abs(fidelity)

    return kernel


class QMLKernel:
    def __init__(self, train_data=None, train_labels=None, test_data=None, test_labels=None, model='svc'):
        self._model = None
        self._is_fitted = None
        self._test_kernel = None
        self._train_kernel = None
        if train_data is None:
            self._train_data, self._train_labels, self._test_data, self._test_labels = ad_hoc_data(training_size=20,
                                                                                                   test_size=5,
                                                                                                   n=2,
                                                                                                   gap=0.3,
                                                                                                   one_hot=False)
        else:
            self._train_data, self._train_labels, self._test_data, self._test_labels = train_data, train_labels, test_data, test_labels
        self._num_qubits = len(self._train_data[0])
        self._feature_map = ZZFeatureMap(feature_dimension=self._num_qubits, reps=2)
        if model == 'svc':
            self._model = SVC(kernel='precomputed')

    def random_data(self, training_size=20, test_size=5, n=2):
        train_data, train_labels, test_data, test_labels = ad_hoc_data(training_size=training_size,
                                                                       test_size=test_size, n=n, gap=0.3)
        self.__init__(train_data, train_labels, test_data, test_labels)

    def run(self):
        kernel = FidelityQuantumKernel(feature_map=self._feature_map)

        self._train_kernel = calculate_kernel(self._feature_map, self._train_data)
        self._test_kernel = calculate_kernel(self._feature_map, self._train_data, self._test_data)

        #self._train_kernel = kernel.evaluate(x_vec=self._train_data)
        #self._test_kernel = kernel.evaluate(x_vec=self._test_data, y_vec=self._train_data)
        self._model.fit(self._train_kernel, self._train_labels)
        self._is_fitted = True

    def show_result(self):
        if not self._is_fitted:
            raise RuntimeError('Model is not fitted yet.')
        score = self._model.score(self._test_kernel, self._test_labels)
        print(f"kernel test score: {score}")

    def generate_qasm3(self):
        return qasm3.dumps(self._feature_map.decompose().assign_parameters(self._train_data[0]))

    def plot_data(self):
        unique_labels = np.unique(np.concatenate((self._train_labels, self._test_labels)))
        markers = ['o', 's', '^', 'P', 'D', 'v', 'h', '<', '>', '*']
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        fig, axdata = plt.subplots(figsize=(15, 5))
        axdata.set_title("Data")
        axdata.set_ylim(0, 2 * np.pi)
        axdata.set_xlim(0, 2 * np.pi)

        for label, marker, color in zip(unique_labels, markers, colors):
            # Plot training data
            axdata.scatter(self._train_data[self._train_labels == label, 0],
                           self._train_data[self._train_labels == label, 1],
                           marker=marker, facecolors='none', edgecolors=color,
                           label=f"Train label {label}")
            # Plot test data
            axdata.scatter(self._test_data[self._test_labels == label, 0],
                           self._test_data[self._test_labels == label, 1],
                           marker=marker, facecolors=color, edgecolors=color,
                           label=f"Test label {label}")

        plt.legend()
        plt.show()
