import sympy as sp

# Define symbolic variables
a, b, c = sp.symbols('a b c')

# Define the 3x3 matrix
U = sp.Matrix([
    [0, -a, -b],
    [a, 0, -c],
    [b, c, 0]
])

# Normalize the matrix
norm = sp.sqrt(b**2 + c**2)
U_normalized = U / norm

# Compute U^T * U
U_T_U = sp.simplify(U_normalized.T * U_normalized)

# Extract diagonal and off-diagonal entries
diagonals = [U_T_U[i, i] for i in range(3)]
off_diagonal_12 = U_T_U[0, 1]

# Display results
print("U^T * U:")
sp.pprint(U_T_U)

print("\nDiagonal Entries:")
sp.pprint(diagonals)

print("\nOff-Diagonal Entry (1,2):")
sp.pprint(off_diagonal_12)
