import numpy as np
from scipy.sparse import lil_matrix

def assemble_matrix(N):
    """
    Assemble the system matrix A for the 2D Poisson equation -Delta U = f
    on a square domain [0,1]x[0,1] with Dirichlet boundary conditions.
    The domain is discretized into (N+1)x(N+1) grid points,
    leading to (N-1)x(N-1) interior points.

    Parameters:
    N (int): Number of intervals along one side of the domain.
             The grid step size h is 1/N.

    Returns:
    scipy.sparse.lil_matrix: The system matrix A.
    """
    # Number of interior points in one dimension
    num_interior_points_1d = N - 1

    # Total number of interior points
    # This will be the size of our matrix A (M x M)
    M = num_interior_points_1d * num_interior_points_1d

    # Initialize a sparse matrix in LIL format (efficient for construction)
    A = lil_matrix((M, M))

    # Calculate grid spacing
    h = 1.0 / N
    h_squared = h**2

    # Loop over all interior grid points (i, j)
    # i represents the x-index, j represents the y-index
    # Note: Our grid points are (x_i, y_j) where i,j range from 0 to N.
    # Interior points (x_i, y_j) have i,j ranging from 1 to N-1.
    for j in range(1, N): # loop over y-indices
        for i in range(1, N): # loop over x-indices

            # Convert (i, j) 2D index to a 1D global index k
            # k = (j_grid_index - 1) * num_interior_points_1d + (i_grid_index - 1)
            # This follows the lexicographical ordering: row by row
            k = (j - 1) * num_interior_points_1d + (i - 1)

            # Apply the 5-point stencil:
            # 4*U_i,j - U_{i+1,j} - U_{i-1,j} - U_{i,j+1} - U_{i,j-1} = h^2 * f_i,j

            # Diagonal term: 4 * U_i,j
            A[k, k] = 4.0

            # Off-diagonal terms (-1 for neighbors)
            # U_{i+1,j} (right neighbor)
            if i + 1 < N: # Check if it's an interior point
                A[k, (j - 1) * num_interior_points_1d + i] = -1.0 # Index for U_{i+1,j}
            # If i+1 == N, this neighbor is on the right boundary, its value is known
            # and is handled by the RHS (assemble_rhs function later).

            # U_{i-1,j} (left neighbor)
            if i - 1 > 0: # Check if it's an interior point
                A[k, (j - 1) * num_interior_points_1d + (i - 2)] = -1.0 # Index for U_{i-1,j}
            # If i-1 == 0, this neighbor is on the left boundary.

            # U_{i,j+1} (top neighbor)
            if j + 1 < N: # Check if it's an interior point
                A[k, j * num_interior_points_1d + (i - 1)] = -1.0 # Index for U_{i,j+1}
            # If j+1 == N, this neighbor is on the top boundary.

            # U_{i,j-1} (bottom neighbor)
            if j - 1 > 0: # Check if it's an interior point
                A[k, (j - 2) * num_interior_points_1d + (i - 1)] = -1.0 # Index for U_{i,j-1}
            # If j-1 == 0, this neighbor is on the bottom boundary.

    return A


def create_grid(N):
    """
    Creates the 2D grid points for the domain [0,1]x[0,1].

    Parameters:
    N (int): Number of intervals along one side of the domain.

    Returns:
    tuple: (x_coords, y_coords) where x_coords and y_coords are 1D arrays
           of length (N+1) representing the grid lines.
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1) # x-coordinates: 0, h, 2h, ..., 1
    y = np.linspace(0, 1, N + 1) # y-coordinates: 0, h, 2h, ..., 1
    return x, y


def assemble_rhs(f_func, g_func, N):
    """
    Assemble the right-hand side vector B for the 2D Poisson equation.

    Parameters:
    f_func (callable): Function f(x, y) representing the source term.
    g_func (callable): Function g(x, y) representing the Dirichlet boundary conditions.
    N (int): Number of intervals along one side of the domain.

    Returns:
    numpy.ndarray: The right-hand side vector B.
    """
    num_interior_points_1d = N - 1
    M = num_interior_points_1d * num_interior_points_1d
    B = np.zeros(M)

    h = 1.0 / N
    h_squared = h**2

    x_grid, y_grid = create_grid(N)

    # Loop over all interior grid points (i, j)
    for j in range(1, N): # loop over y-indices
        for i in range(1, N): # loop over x-indices

            k = (j - 1) * num_interior_points_1d + (i - 1) # Global 1D index for U_i,j

            # Initialize RHS with h^2 * f(x_i, y_j)
            B[k] = h_squared * f_func(x_grid[i], y_grid[j])

            # Add contributions from known boundary values
            # U_{i-1,j} term: if i-1 == 0 (left boundary)
            if i == 1:
                B[k] += g_func(x_grid[0], y_grid[j])

            # U_{i+1,j} term: if i+1 == N (right boundary)
            if i == N - 1:
                B[k] += g_func(x_grid[N], y_grid[j])

            # U_{i,j-1} term: if j-1 == 0 (bottom boundary)
            if j == 1:
                B[k] += g_func(x_grid[i], y_grid[0])

            # U_{i,j+1} term: if j+1 == N (top boundary)
            if j == N - 1:
                B[k] += g_func(x_grid[i], y_grid[N])

    return B


if __name__ == "__main__":
    print("Testing assemble_matrix function...")

    # Test for a small N, e.g., N=3
    # This means 2x2 = 4 interior points
    # Matrix A should be 4x4
    N_test = 3
    A_test = assemble_matrix(N_test)
    print(f"\nMatrix A for N={N_test} (h=1/{N_test}):")
    print(A_test.toarray()) # Convert to dense array for easy viewing

    # Expected A for N=3 (2x2 interior points)
    # [ (1,1)  (2,1)  (1,2)  (2,2) ] -> Indices (i,j)
    # k = 0,   1,     2,     3
    #
    # 4*U_11 - U_21 - U_12 = h^2 f_11 + U_01 + U_10 (boundary terms)
    # 4*U_21 - U_11 - U_22 = h^2 f_21 + U_31 + U_20 (boundary terms)
    # 4*U_12 - U_22 - U_11 = h^2 f_12 + U_02 + U_13 (boundary terms)
    # 4*U_22 - U_12 - U_21 = h^2 f_22 + U_32 + U_23 (boundary terms)
    #
    # Expected matrix (ignoring boundary terms in f for now):
    # [[ 4, -1, -1,  0],
    #  [-1,  4,  0, -1],
    #  [-1,  0,  4, -1],
    #  [ 0, -1, -1,  4]]


    # Test for a larger N, e.g., N=4
    # This means 3x3 = 9 interior points
    # Matrix A should be 9x9
    N_test_large = 4
    A_large = assemble_matrix(N_test_large)
    print(f"\nMatrix A for N={N_test_large} (h=1/{N_test_large}):")
    print(A_large.toarray()) # Might be too big to print cleanly

    print(f"Shape of A for N={N_test_large}: {A_large.shape}")
    print(f"Number of non-zero elements: {A_large.nnz}")

    # You can also check specific elements
    # For N=4, M=(4-1)*(4-1) = 3*3 = 9
    # Point (1,1) -> k=0: neighbors are (2,1), (1,2)
    print(f"A[0,0] (diagonal for U_11): {A_large[0,0]}") # Should be 4
    print(f"A[0,1] (U_21 neighbor of U_11): {A_large[0,1]}") # Should be -1
    print(f"A[0,3] (U_12 neighbor of U_11): {A_large[0,3]}") # Should be -1

    # Point (2,2) -> k = (2-1)*3 + (2-1) = 1*3+1 = 4
    # Neighbors (1,2), (3,2), (2,1), (2,3)
    print(f"A[4,4] (diagonal for U_22): {A_large[4,4]}") # Should be 4
    print(f"A[4,3] (U_12 neighbor): {A_large[4,3]}") # Should be -1
    print(f"A[4,5] (U_32 neighbor): {A_large[4,5]}") # Should be -1
    print(f"A[4,1] (U_21 neighbor): {A_large[4,1]}") # Should be -1
    print(f"A[4,7] (U_23 neighbor): {A_large[4,7]}") # Should be -1

    print("\nTest completed.")