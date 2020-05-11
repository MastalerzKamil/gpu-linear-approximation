from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import pyopencl as cl


def jacobi(A, b, N=25, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = cl.zeros_type()
        # x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        x = (b - dot(R, x)) / D
    return x


def read_from_file(filename):
    with open(filename) as textFile:
        lines = [line.split() for line in textFile]
        return lines


def main():
    A = array([[2.0, 1.0], [5.0, 7.0]])
    b = array([11.0, 13.0])
    guess = array([1.0, 1.0])
    test = read_from_file("dane2.txt")
    print(test)

    sol = jacobi(A, b, N=25, x=guess)

    print("A:")
    pprint(A)

    print("b:")
    pprint(b)

    print("x:")
    pprint(sol)


if __name__ == "__main__":
    main()
