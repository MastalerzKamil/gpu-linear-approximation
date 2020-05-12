#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void interpolate(__global const double *dataX,  __global const double *dataY, __global int *size, __global double *x,  __global double *result){
    double sum = 0, temp;
    int n = *size;
    for (int i = 0; i < n; ++i) {
        temp = 1;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                temp *= (x[get_global_id(0)] - dataX[j]) / (dataX[i] - dataX[j]);
            }    
        }
        sum += dataY[i] * temp;
    }
    result[get_global_id(0)] = sum;
}