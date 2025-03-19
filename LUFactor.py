# use Doolittle's algorithm for LU factorization, with 1's on the diagonal of L.
# time complexity: O(n^3)
# reference: https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/

import numpy as np

def compareMatrix(A,B): # A and B are same square dimensions
    n = A.shape[0]
    for i in range (n):
        for j in range (n):
            if A[i, j] != B[i, j]: return False
    
    return True
    
def matrixMultiply(A, B): 
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

# check if a matrix is lower triangular: all entries for i < j = 0
def isLowerTriangular(A): # A is square
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Check elements above the diagonal
            if A[i, j] != 0:
                return False
    return True


def isUpperTriangular(A):
    n = A.shape[0]
    for i in range(1, n):
        for j in range(i):  # Check elements below the diagonal
            if A[i, j] != 0:
                return False
    return True

def check1sOnDiagonal(A): # A is square
    n = A.shape[0]
    for i in range (n):
        if A[i,i] != 1:
            return False
    return True

def LUFactor(A):
    n = A.shape[0]
    L = np.eye(n)  # L starts as n identity matrix
    U = np.zeros((n, n))  # U starts as n zero matrix

    for i in range(n):  

        for j in range(i, n):  
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        for j in range(i+1, n):  
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U


# driver:
A = np.array([[4, 12, -16], [12, 37, -43], [-16, 43, 98]], dtype=float)
L, U = LUFactor(A)
print("A=\n", A)
# make sure L is lower triangular:
if (isLowerTriangular(L) and isUpperTriangular(U)):
    if(check1sOnDiagonal(L)):
        # multiply L and LT and compare to A to verify factorization accuracy.
        if (compareMatrix(A, matrixMultiply(L, U)) == True):
            print("\n\nSuccess: L*U=A")
            print("L=\n", L)
            print("U=\n", U)
        else: print ("Failed: multiplication test")
    else: print("Failed: L does not have 1's on diagonal")
else: print("Failed: L is not lower triangular or U is not upper triangular")
