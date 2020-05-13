#!/usr/bin/python

import getopt
import sys

from numpy import zeros, diag, diagflat, dot

from FileReader import FileReader
from Lagrange import Lagrange

"""
def jacobi(A, b, N=25, x=None):
    # Solves the equation Ax=b via the Jacobi iterative method.
    # Create an initial guess if needed
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        x = (b - dot(R, x)) / D
    return x
"""

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print("Welcome in function approximation program")
        print("Parameters: \n-i --input <input file> \n-o --output <output file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Welcome in function approximation program")
            print("Parameters: \n-i --input <input file> \n-o --output <output file>")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
    print('Input file is "', inputfile)
    print('Output file is "', outputfile)
    x_sample = [0, 1, 2, 3, 4, 5, 6]
    y_sample = [0, 1, 4, 9, 16, 25, 36]
    file_data = FileReader()
    file_data.read_from_file("dane_test.txt")
    example = Lagrange(x_sample, y_sample)
    example.execute("lagrangeInterpolate.cl")
    example.write_to_file("result.txt")


if __name__ == "__main__":
    main(sys.argv[1:])
