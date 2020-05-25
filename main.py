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


class PyOpenClFactory:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    """
        self.ctx = cl.Context(
            dev_type=cl.device_type.ALL,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])
    """

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

    def allocate_buff_lagrange(self, x_vector, y_vector, data_size, x, host_dest):
        mf = cl.mem_flags

        self.x_vector_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_vector)
        self.y_vector_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_vector)
        self.data_size_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_size)
        self.x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, host_dest.nbytes)

    def __del__(self):
        self.queue.finish()


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


class Lagrange:
    def __init__(self, x_data, y_data):
        self.data_size = np.array([len(x_data)])
        self.x_vector = np.array(x_data).astype(np.double)
        self.y_vector = np.array(y_data).astype(np.double)
        self.delta_x = self.calculate_delta_x()
        self.start_value = self.get_first_x_vector_element()
        self.last_value = self.get_last_x_vector_element()
        self.result_x_host_data = []
        self.result_y_host_data = []

    def get_last_x_vector_element(self):
        return self.x_vector[len(self.x_vector) - 1]

    def get_first_x_vector_element(self):
        return self.x_vector[0]

    def calculate_delta_x(self):
        return self.x_vector[1] - self.x_vector[0]

    def execute(self, program_filename):
        for x in np.arange(self.start_value, self.last_value, self.delta_x):
            self.result_x_host_data.append(x)
            result_y_temp = np.array([1.0])

            cl_instance = PyOpenClFactory()
            cl_instance.allocate_buff_lagrange(self.x_vector, self.y_vector, self.data_size, x, result_y_temp)
            cl_instance.loadProgram(program_filename)
            cl_instance.program.interpolate(cl_instance.queue, self.x_vector.shape, None, cl_instance.x_vector_buf,
                                            cl_instance.y_vector_buf, cl_instance.data_size_buf, cl_instance.x_buf,
                                            cl_instance.dest_buf)
            cl.enqueue_copy(cl_instance.queue, result_y_temp, cl_instance.dest_buf).wait()
            # del cl_instance
            self.result_y_host_data.append(result_y_temp[0])

    def show_results(self):
        print("\nx input", self.x_vector)
        print("y input", self.y_vector)
        print("result: ", self.result_y_host_data)
        plt.plot(self.result_x_host_data, self.result_y_host_data, self.x_vector, self.y_vector)
        plt.show()

    def write_to_file(self, filename):
        n = len(self.result_x_host_data)
        connected_result = []

        for i in range(n):
            connected_result.append([self.result_x_host_data[i], self.result_y_host_data[i]])
        result_matrix = np.array(connected_result)
        # print("interpolated")
        # print(result_matrix)
        np.savetxt(filename, result_matrix)


class CubicSplines:
    def __init__(self, x_data, y_data):
        self.x_vector = x_data
        self.y_vector = y_data

    def prepare_new_x(self, x):
        delta_x = abs(x[1]-x[0]) # delta can't be negative
        last_x = x[len(x)-1]
        new_x = []
        for i in np.arange(x[0], last_x, delta_x):
            new_x.append(i)
        return np.array(new_x)

    def calculate_cubic(self, program_filename):
        """
        Interpolate a 1-D function using cubic splines.
          x0 : a float or an 1d-array
          x : (N,) array_like
              A 1-D array of real/complex values.
          y : (N,) array_like
              A 1-D array of real values. The length of y along the
              interpolation axis must be equal to the length of x.

        Implement a trick to generate at first step the cholesky matrice L of
        the tridiagonal matrice A (thus L is a bidiagonal matrice that
        can be solved in two distinct loops).

        additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
        """
        self.x_vector = np.asfarray(self.x_vector)
        self.y_vector = np.asfarray(self.y_vector)

        self.new_x = self.prepare_new_x(self.x_vector)

        # check if sorted
        if np.any(np.diff(self.x_vector) < 0):
            indexes = np.argsort(self.x_vector)
            self.x_vector = self.x_vector[indexes]
            self.y_vector = self.y_vector[indexes]
        size = len(self.x_vector)

        xdiff = np.diff(self.x_vector).astype(np.double)
        ydiff = np.diff(self.y_vector).astype(np.double)

        # allocate buffer matrices
        Li = np.empty(size).astype(np.double)
        Li_1 = np.empty(size - 1).astype(np.double)
        z = np.empty(size).astype(np.double)

        # fill diagonals Li and Li-1 and solve [L][y] = [B]
        Li[0] = sqrt(2 * xdiff[0])
        Li_1[0] = 0.0
        B0 = 0.0  # natural boundary
        z[0] = B0 / Li[0]

        Bi = np.zeros((1)).astype(np.double)

        for i in range(1, size - 1, 1):
            Li_1[i] = xdiff[i - 1] / Li[i - 1]
            Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
            Bi[0] = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
            z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]
        i = size - 1
        Li_1[i - 1] = xdiff[-1] / Li[i - 1]
        Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
        Bi = 0.0  # natural boundary
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        # solve [L.T][x] = [y]
        i = size - 1
        z[i] = z[i] / Li[i]
        for i in range(size - 2, -1, -1):
            z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

        # find index
        index = self.x_vector.searchsorted(self.new_x)
        # np.clip(index, 1, size-1, index)

        xi1, xi0 = self.x_vector[index], self.x_vector[index - 1]
        yi1, yi0 = self.y_vector[index], self.y_vector[index - 1]
        zi1, zi0 = z[index], z[index - 1]
        hi1 = xi1 - xi0  # TODO parallelize

        cl_cubic_splines_instance = PyOpenClFactory()
        mf = cl.mem_flags

        # allocating buffers for cubic calculations
        zi0_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zi0)
        hi1_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hi1)
        xi1_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xi1)
        x0_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.new_x)
        xi0_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xi0)
        yi1_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yi1)
        zi1_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zi1)
        yi0_buff = cl.Buffer(cl_cubic_splines_instance.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yi0)
        res_buf = cl.Buffer(cl_cubic_splines_instance.ctx, mf.WRITE_ONLY, yi0.nbytes)

        cl_cubic_splines_instance.loadProgram(program_filename)
        cl_cubic_splines_instance.program.calculate_cubic(cl_cubic_splines_instance.queue, np.array(zi0).shape, None,
                                                          zi0_buff, hi1_buff, xi1_buff, x0_buff, xi0_buff, yi1_buff,
                                                          zi1_buff, yi0_buff, res_buf)

        # calculate cubic
        self.result_y = np.empty(len(self.new_x)).astype(np.double)
        cl.enqueue_copy(cl_cubic_splines_instance.queue, self.result_y, res_buf).wait()


def main(argv):
    inputfile = ''
    outputfile = ''
    env_device = '0'
    method = []
    try:
        opts, args = getopt.getopt(argv, "hi:o:d:sl", ["input=", "output=","device="])
    except getopt.GetoptError:
        print("Welcome in function approximation program")
        print("Parameters: \n"
              "-i --input <input file> \n"
              "-o --output <output file> \n"
              "-d --device <device value>\n"
              "0 - intel\n"
              "1- CUDA")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Welcome in function approximation program")
            print("Parameters: \n"
                  "-i --input <input file> \n"
                  "-o --output <output file> \n"
                  "-d --device <device value>\n"
                  "0 - intel\n"
                  "1- CUDA")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
        elif opt in ("-d", "--device"):
            env_device = arg
        elif opt in ("-s", "--splines"):
            method.append("splines")
        elif opt in ("-l", "--lagrange"):
            method.append("lagrange")

    os.environ["PYOPENCL_CTX"] = env_device

    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    device= {
            '0': "Intel",
            '1': "CUDA"
            }
    print(device.get(env_device,"Invalid device"))
    x_sample = [0, 1, 2, 3, 4, 5, 6]
    y_sample = [0, 1, 4, 9, 16, 25, 36]
    file_data = FileReader()
    file_data.read_from_file(inputfile)

    example_spline = CubicSplines(file_data.x_vector, file_data.y_vector)
    time_start = time.time()
    print("executing Splines")
    example_spline.calculate_cubic("cubic-splines-kernel.cl")
    time_end = time.time()
    print("--- %s seconds ---" % (time_end - time_start))
    file_data.write_to_file(example_spline.new_x, example_spline.result_y, outputfile)

"""
    example = Lagrange(file_data.x_vector, file_data.y_vector)
    time_start = time.time()
    print("executing Lagrange")
    example.execute("lagrangeInterpolate.cl")
    time_end = time.time()
    print("--- %s seconds ---" % (time_end - time_start))
    example.write_to_file(outputfile)
    file_data.write_to_file(example.result_x_host_data, example.result_y_host_data, outputfile)
"""


if __name__ == "__main__":
    main(sys.argv[1:])