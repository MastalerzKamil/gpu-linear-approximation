import numpy as np
from math import sqrt
import os
import pyopencl as cl


def loadProgram(ctx, filename):
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    return cl.Program(ctx, fstr).build()


def cubic_interp1d(x0, x, y):
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
    x = np.asfarray(x)
    y = np.asfarray(y)

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]
    size = len(x)

    xdiff = np.diff(x).astype(np.double)
    ydiff = np.diff(y).astype(np.double)

    # allocate buffer matrices
    Li = np.empty(size).astype(np.double)
    Li_1 = np.empty(size-1).astype(np.double)
    z = np.empty(size).astype(np.double)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    Bi = np.zeros((1)).astype(np.double)


    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi[0] = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]
    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = x.searchsorted(x0)
    # np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0  # TODO parallelize

    ctx = cl.create_some_context()
    mf = cl.mem_flags
    queue = cl.CommandQueue(ctx)

    zi0_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zi0)
    hi1_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hi1)
    xi1_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xi1)
    x0_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0)
    xi0_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xi0)
    yi1_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yi1)
    zi1_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=zi1)
    yi0_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yi0)
    res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, yi0.nbytes)

    program = loadProgram(ctx, "cubic-splines-kernel.cl")
    program.calculate_cubic(queue, np.array(zi0).shape, None, zi0_buff, hi1_buff, xi1_buff, x0_buff, xi0_buff, yi1_buff,
                            zi1_buff, yi0_buff, res_buf)

    # calculate cubic
    f0 = np.empty(len(x0)).astype(np.double)
    cl.enqueue_copy(queue, f0, res_buf).wait()
    return f0

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # os.environ["PYOPENCL_CTX"] = '1'
    x = np.linspace(0, 10, 11)
    y = np.sin(x)
    plt.scatter(x, y)

    x_new = np.linspace(0, 10, 201)
    plt.plot(x_new, cubic_interp1d(x_new, x, y))
    print(x_new, cubic_interp1d(x_new,x, y))
    # plt.show()