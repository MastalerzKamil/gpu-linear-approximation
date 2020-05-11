import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


def calculate_l_value(iteration, points_matrix, x_value):
    l_value = 1
    n = len(points_matrix)
    for j in range(n):
        if j == iteration:
            continue
        xj, yj = points_matrix[j]
        l_value *= float(x_value - xj) / float(points_matrix[iteration][0] - xj)
    return l_value


# points is a 2d array with [x,y] per each record
def lagrange(points_matrix, x_value):
    result = 0
    n = len(points_matrix)
    for i in range(n):
        xi, yi = points_matrix[i]
        result += yi * calculate_l_value(i, points_matrix, x_value)
    return result


def execute_method(input_matrix, method, out_size):
    out_matrix = []
    for i in range(out_size):
        if method == "-l":
            single_record = [i, lagrange(input_matrix, i)]
            out_matrix.append(single_record)
    return out_matrix


def cl_lagrange(context, x_value):
    return cl.Program(context,"""
    __kernel void calculateLagrange(__global const float4 *arrayAsOpenCLType, int x, int arrayLength) {
        float result=0;
        for(int i=0;i<arrayLength;i++) {
            
        }
    }
    """).build()


def calculateExample(context):
    return cl.Program(context, """
        __kernel void multiplyByTwo(__global const float4 *arrayAsOpenCLType, __global float4 *resultAsOpenCLType) {
            int gid = get_global_id(0);

            float4 vector = arrayAsOpenCLType[gid];
            resultAsOpenCLType[gid] =  vector * (float) 2.0;
        }
        """).build()


def main():
    # Context
    ctx = cl.create_some_context()
    # Create queue
    queue = cl.CommandQueue(ctx)

    someArray = np.array([
        [1, 1], [3, 9], [5, 25], [6, 36], [8, 64], [10, 100], [12, 144]
    ]).astype(np.float32)

    print("\nInput:")
    print(someArray)
    print("------------------------------------")

    # Get mem flags
    mf = cl.mem_flags

    # Create a read-only buffer on device and copy 'someArray' from host to device
    arrayAsOpenCLType = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=someArray)

    # Create a write-only buffer to get the result from device
    resultAsOpenCLType = cl.Buffer(ctx, mf.WRITE_ONLY, someArray.nbytes)

    # Creates a kernel in context
    program = calculateExample(ctx)

    # Execute
    program.multiplyByTwo(queue, someArray.shape, None, arrayAsOpenCLType, resultAsOpenCLType)

    # Creates a buffer for the result (host memory)
    result = np.empty_like(someArray)

    # Copy the results from device to host
    cl.enqueue_copy(queue, result, resultAsOpenCLType)

    print("------------------------------------")
    print("Output")
    # Show the result
    print(result)
'''
    my_data = np.array([[1, 1], [3, 9], [5, 25], [6, 36], [8, 64], [10, 100], [12, 144]]).astype(np.float32)
    out_data = execute_method(my_data, "-l", 13)
    out_data = np.array(out_data)

    print(out_data)
    plt.plot(my_data[:, 0], my_data[:, 1], 'r', out_data[:, 0], out_data[:, 1], 'b')
    plt.show()
'''


if __name__ == "__main__":
    main()

'''

    __kernel void executeChosenMethod(__global float4 *inputMatrix, char *method, int outSize) {
        int gid = get_global_id(0);
    }
'''