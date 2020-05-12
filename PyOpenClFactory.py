# Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt


class Lagrange:
    def __init__(self, x_data, y_data, result_size):
        self.result_size = result_size
        self.data_size = np.array([len(x_data)])
        self.x_vector = np.array(x_data).astype(np.float32)
        self.y_vector = np.array(y_data).astype(np.float32)
        self.delta_x = self.calculate_delta_x()
        self.start_value = self.get_first_x_vector_element()
        self.last_value = self.get_last_x_vector_element()

    def get_last_x_vector_element(self):
        return self.x_vector[len(self.x_vector) - 1]

    def get_first_x_vector_element(self):
        return self.x_vector[0]

    def calculate_delta_x(self):
        return self.x_vector[1] - self.x_vector[0]

    def execute(self, program_filename):
        result_x_host_data = []
        result_y_host_data = []

        for x in np.arange(self.start_value, self.last_value, self.delta_x):
            result_x_host_data.append(x)
            result_y_temp = np.array([0.0])

            cl_instance = CL()
            cl_instance.popCorn(self.x_vector, self.y_vector, self.data_size, x, result_y_temp)
            cl_instance.loadProgram(program_filename)
            cl_instance.program.interpolate(cl_instance.queue, self.x_vector.shape, None, cl_instance.x_vector_buf,
                                            cl_instance.y_vector_buf, cl_instance.data_size_buf, cl_instance.x_buf,
                                            cl_instance.dest_buf)
            cl.enqueue_copy(cl_instance.queue, result_y_temp, cl_instance.dest_buf).wait()
            print(x, result_y_temp[0])
            result_y_host_data.append(result_y_temp[0])

        print("\na", self.x_vector)
        print("b", self.y_vector)
        print("result: ", result_y_host_data)
        plt.plot(result_x_host_data, result_y_host_data, self.x_vector, self.y_vector)
        plt.show()


class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

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


if __name__ == "__main__":
    x_sample = [1, 2, 3, 4, 5, 6]
    y_sample = [1, 4, 9, 16, 25, 36]
    example = Lagrange(x_sample, y_sample, 12)
    example.execute("lagrangeInterpolate.cl")
