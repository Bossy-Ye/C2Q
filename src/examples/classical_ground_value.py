import numpy as np

# Define the Pauli matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Define the Kronecker products
I_I = np.kron(I, I)
I_I_I = np.kron(I_I, I)
I_I_I_I_I_I = np.kron(I_I_I, I_I_I)
I_I_I_I_I_I_I = np.kron(I_I_I_I_I_I, I)

X_X = np.kron(X, X)
X_X_X = np.kron(X_X, X)
X_X_X_X_X_X = np.kron(X_X_X, X_X_X)
X_X_X_X_X_X_X = np.kron(X_X_X_X_X_X, X)

Y_Y = np.kron(Y, Y)
Y_Y_Y = np.kron(Y_Y, Y)
Y_Y_Y_Y_Y_Y = np.kron(Y_Y_Y, Y_Y_Y)
Y_Y_Y_Y_Y_Y_Y = np.kron(Y_Y_Y_Y_Y_Y, Y)

Z_Z = np.kron(Z, Z)
Z_Z_Z = np.kron(Z_Z, Z)
Z_Z_Z_Z_Z_Z = np.kron(Z_Z_Z, Z_Z_Z)
Z_Z_Z_Z_Z_Z_Z = np.kron(Z_Z_Z_Z_Z_Z, Z)

# Define the observable matrix
O1 = (2 * I_I_I_I_I_I_I + 5 * X_X_X_X_X_X_X
      - 4 * Y_Y_Y_Y_Y_Y_Y - 3 * Z_Z_Z_Z_Z_Z_Z
      - 8 * Y_Y_Y_Y_Y_Y_Y)
print(O1)
# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(O1)

print("Eigenvalues:", eigenvalues)