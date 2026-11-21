import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    N = len(A)
    M = len(A[0])
    ans = np.zeros((M,N), dtype = float)
    for i in range(N):
        for j in range(M):
            ans[j][i] = A[i][j]

    return ans