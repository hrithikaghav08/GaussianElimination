#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes

gauss_library_path = './libgauss.so'

def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)


def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

def plu_python(A):
    """Python implementation of PA=LU decomposition with partial pivoting."""
    n = len(A)
    P = list(range(n))

    for k in range(n):
        # Find pivot
        pivot = max(range(k, n), key=lambda i: abs(A[i][k]))
        if A[pivot][k] == 0:
            raise ValueError("Singular matrix")
        # Swap rows
        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
            P[k], P[pivot] = P[pivot], P[k]
        # Perform LU decomposition
        for i in range(k+1, n):
            A[i][k] /= A[k][k]
            for j in range(k+1, n):
                A[i][j] -= A[i][k] * A[k][j]
    
    L, U = unpack(A)
    return P, L, U

def plu_c(A):
    """C implementation of PA=LU decomposition."""
    lib = ctypes.CDLL(gauss_library_path)
    n = len(A)
    flat_array_2d = [item for row in A for item in row]
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)
    
    # Create an array for the permutation vector P
    P = (ctypes.c_int * n)()

    # Define the function signature
    lib.plu.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
    
    # Call the C function
    lib.plu(n, c_array_2d, P)
    
    # Convert the modified A matrix and P array back to Python structures
    modified_array_2d = [[c_array_2d[i * n + j] for j in range(n)] for i in range(n)]
    P_list = list(P)
    
    L, U = unpack(modified_array_2d)
    return P_list, L, U

def plu(A, use_c=False):
    """Perform PA=LU decomposition using either Python or C."""
    if use_c:
        return plu_c(A)
    else:
        return plu_python(A)

if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()

    L, U = lu(A, use_c = False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
