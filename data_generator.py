#!/usr/bin/python

import getopt
import os
import sys
from math import sqrt

from numpy import zeros, diag, diagflat, dot

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time

class FileReader:
    def __init__(self):
        self.x_vector = []
        self.y_vector = []

    def read_from_file(self, filename):
        file = open(filename)
        for line in file:
            fields = line.split(" ")
            self.x_vector.append(float(fields[0]))
            self.y_vector.append(float(fields[1]))

    def get_x_vector(self):
        return self.x_vector

    def get_y_vector(self):
        return self.y_vector

    def write_to_file(self, result_x, result_y, filename):
        n = len(result_x)
        connected_result = []

        for i in range(n):
            connected_result.append([result_x[i], result_y[i]])
        result_matrix = np.array(connected_result)
        # print("interpolated")
        # print(result_matrix)
        np.savetxt(filename, result_matrix)


class DatasetGenerator:
    def __init__(self, delta, start, end):
        self.x_vector = []
        self.y_vector = []
        self.delta = delta
        self.start = start
        self.end = end

    def generate(self, low, high):
        for x in np.arange(self.start, self.end, self.delta):
            self.x_vector.append(x)
        y_length = len(self.x_vector)
        self.y_vector = high + np.random.sample(y_length) * low

    def save_to_file(self, filename):
        n = len(self.x_vector)
        connected_result = []

        for i in range(n):
            connected_result.append([self.x_vector[i], self.y_vector[i]])
        np.savetxt(filename, np.array(connected_result))

def main(argv):
    delta = 0.1
    a0 = 200
    a1 = 10000
    start = 0
    end = 150
    output = 'data5.txt'
    try:
        opts, args = getopt.getopt(argv, "d:a0:a1:s:e:o:", ["delta=", "amplitude0=","amplitude1=",
                                                            "start=","end=", "output="])
    except getopt.GetoptError:
        print("problem with parapeters")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-d':
            print("delta chosen")
            delta=float(arg)
            sys.exit()
        elif opt == "-a0":
            a0 = float(arg)
        elif opt == "-a1":
            a1 = float(arg)
        elif opt == "-s":
            start = float(arg)
        elif opt == "-e":
            end = float(arg)
        elif opt == "-o":
            output = arg


    dataset = DatasetGenerator(delta, start, end)
    dataset.generate(a0, a1)
    dataset.save_to_file(output)


if __name__ == '__main__':
    main(sys.argv[1:])