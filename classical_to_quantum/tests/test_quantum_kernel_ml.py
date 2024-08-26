from classical_to_quantum.qasm_generate import QASMGenerator
classical_code = """
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Only take the first two features for simplicity
X = X[:, :2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier with an RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Train the classifier
svm_rbf.fit(X_train, y_train)

# Make predictions
y_pred = svm_rbf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
"""
generator=QASMGenerator()
qasm = generator.qasm_generate(classical_code, verbose=True)