from typing import List

import numpy as np


def seidel_algorithm(A: List[List] or np.ndarray):
    """Compute the shortest-paths lengths."""
    A = np.asarray(A)
    n = len(A)

    # If A is fully-connected return A
    if all(A[i][j] for i in range(n) for j in range(n) if i != j):
        return A

    # Degree of G to its adj matrix
    adj_matrices = {1: A}
    # n'
    n1 = 1
    An_with_ones = A.copy()
    # Set all diagonals to one
    for i in range(n):
        An_with_ones[i][i] = 1
    # A -> A^2 -> A^4 .... A^n' where n' >= n
    while n1 < n:
        An_with_ones = An_with_ones @ An_with_ones
        n1 *= 2
        # Turn all non-zero values to 1 and all diagonal elements to 0
        adj_matrices[n1] = np.asarray(
            [[1 if i != j and (An_with_ones[i][j] > 0) else 0 for j in range(n)] for i in range(n)])

    # An with zeros on diagonals = Distance matrix of G^n1
    # Matrix of distances D of G^2n
    D_g2 = adj_matrices[n1]
    # Matrix D of G^n
    D_g = np.ndarray(shape=[n, n])

    # D_g^n -> D_g^(n/2) -> ... -> D_g^(2) -> D_g
    while n1 > 1:
        # Adg matrix of G^(n1 / 2)
        adj_matrix = adj_matrices[n1 // 2]
        # D_g2 * adj matrix
        DA = D_g2 @ adj_matrix
        for u in range(n):
            for v in range(n):
                if u == v:
                    D_g[u][v] = 0
                    continue
                # D_g2(u, v) * degree of vertex u in original G
                compare_val = D_g2[u][v] * sum(adj_matrix[v])
                D_g[u][v] = 2 * D_g2[u][v] if DA[u][v] >= compare_val else 2 * D_g2[u][v] - 1
        n1 /= 2
        D_g2 = D_g
        # print(D_g)
    #  Returns D of original graph
    return D_g


if __name__ == '__main__':
    A = np.asarray([
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0]])
    print("Input:")
    print(A)
    print("Output:")
    print(seidel_algorithm(A))
