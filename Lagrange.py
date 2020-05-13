import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

from FileReader import FileReader

from PyOpenClFactory import PyOpenClFactory


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
            print(x, result_y_temp[0])
            self.result_y_host_data.append(result_y_temp[0])

        print("\na", self.x_vector)
        print("b", self.y_vector)
        print("result: ", self.result_y_host_data)
        plt.plot(self.result_x_host_data, self.result_y_host_data, self.x_vector, self.y_vector)
        plt.show()

    def write_to_file(self,filename):
        n = len(self.result_x_host_data)
        result_matrix = []

        for i in range(n):
            result_matrix.append([self.result_x_host_data[i], self.result_y_host_data[i]])
        result_matrix = np.array(result_matrix)
        print(result_matrix)
        np.savetxt(filename, result_matrix)


if __name__ == "__main__":
    x_sample = [0, 1, 2, 3, 4, 5, 6]
    y_sample = [0, 1, 4, 9, 16, 25, 36]
    file_data = FileReader()
    file_data.read_from_file("dane_test.txt")
    example = Lagrange(file_data.get_x_vector(), file_data.get_y_vector())
    example.execute("lagrangeInterpolate.cl")
    example.write_to_file("result.txt")
