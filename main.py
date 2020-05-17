#!/usr/bin/python

import getopt
import sys

from numpy import zeros, diag, diagflat, dot

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time


class PyOpenClFactory:
    def __init__(self):
        platforms = cl.get_platforms()  # Select the first platform [0]
        if not platforms:
            raise EnvironmentError('No openCL platform (or driver) available.')

        devices = platforms[0].get_devices()
        self.ctx = cl.Context([devices[0]])
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

    def popCorn(self, x_vector, y_vector, data_size, x, host_dest):
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
            cl_instance.popCorn(self.x_vector, self.y_vector, self.data_size, x, result_y_temp)
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
        np.savetxt(filename, result_matrix,)


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["input=", "output="])
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
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    x_sample = [0, 1, 2, 3, 4, 5, 6]
    y_sample = [0, 1, 4, 9, 16, 25, 36]
    file_data = FileReader()
    file_data.read_from_file(inputfile)
    example = Lagrange(file_data.x_vector, file_data.y_vector)
    time_start = time.time()
    print("executing Lagrange")
    example.execute("lagrangeInterpolate.cl")
    time_end = time.time()
    print("--- %s seconds ---" % (time_end - time_start))
    example.write_to_file(outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
