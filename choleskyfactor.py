# cholskey decomposition: "Every symmetric, positive definite matrix A can be decomposed into a product of a unique lower triangular matrix L and its transpose: 
# A = L*LT "
# reference: https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/

import numpy as np

def transpose(A): # helper function to compute transpose of L
    rows, cols = A.shape
    AT = np.empty((cols, rows), dtype=A.dtype)

    for i in range (rows):
        for j in range(cols):
            AT[j,i] = A[i,j]

    return AT

def compareMatrix(A,B): # A and B are same dimensions
    n = A.shape[0]
    for i in range (n):
        for j in range (n):
            if A[i, j] != B[i, j]:
                return False
    
    return True
    
def matrixMultiply(A, B): # to verify choleskyFactor()
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def checkSymmetric(A):
    AT = transpose(A)
    for i in range (A.shape[0]):
        for j in range(A.shape[0]):
            if A[i,j] != AT[i,j]:
                return False
    return True

# check if a matrix is lower triangular: all entries for i < j = 0
def isLowerTriangular(A): # A is square
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Check elements above the diagonal
            if A[i, j] != 0:
                return False
    return True


def choleskyFactor(A): # Precondition: A is symmetric and positive definite.

    n = A.shape[0]

    if (checkSymmetric(A)):
        j=0
        L = np.zeros((n, n))  # L starts as an n zero matrix

        for i in range (n):

            if A[i, j]:
            # diagonal elements found with sqrt -- ensures a positive diagonal.
                L[i,i] = np.sqrt(A[i,i] - sum(L[i,k] ** 2 for k in range(i)))

            # compute lower triangular part -- above diagonal remains 0.
            for j in range(i+1, n):
                L[j, i] = (A[j, i] - sum(L[j, k] * L[i, k] for k in range(i))) / L[i, i]
        
        return L, transpose(L)

    else: 
        print("A is not symmetric")
        return np.zeros((n,n)), np.zeros((n,n))


# driver:
A = np.array([[4, 2, 2], [2, 5, 3], [2, 3, 6]], dtype=float)
L, LT = choleskyFactor(A)
print("A=\n", A)
# make sure L is lower triangular:
if (isLowerTriangular(L) == True):
    # multiply L and LT and compare to A to verify factorization accuracy.
    if (compareMatrix(A, matrixMultiply(L, LT)) == True):
        print("\n\nSuccess: L*LT=A")
        print("L=\n", L)
        print("LT=\n", LT)
    else: print ("Failed: multiplication test")
else: print("Failed: L is not lower triangular")
     