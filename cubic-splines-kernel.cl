#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void generate_hi1(__global const double* xi1, __global const double* xi0, __global double* res_hi1)
{
    int gid = get_global_id(0);
    res_hi1[gid]=xi1[gid]-xi0[gid];
}

__kernel void calculate_cubic(__global const double* zi0, __global const double* hi1,
__global const double* xi1, __global const double* x0, __global const double* xi0,
__global const double* yi1, __global const double* zi1, __global const double* yi0,
__global double* res)
{
    int gid = get_global_id(0);
    res[gid]=zi0[gid]/(6*hi1[gid])*pow((xi1[gid]-x0[gid]),3) + 
        zi1[gid]/(6*hi1[gid])*pow((x0[gid]-xi0[gid]),3) + 
        (yi1[gid]/hi1[gid] - zi1[gid]*hi1[gid]/6)*(x0[gid]-xi0[gid]) +
        (yi0[gid]/hi1[gid] - zi0[gid]*hi1[gid]/6)*(xi1[gid]-x0[gid]);
}
