#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*
__kernel void diagonals_loop(__global const double* Li_1, __global double* Li, const unsigned int size, __global const double* xdiff,
				      __global const double* ydiff, __global double* z,
                      __global double* res_Li_1, __global double* res_Li, __global double* Bi, __global double* res_z)
{
    int n = size;
    int gid = get_global_id(0);
    
    if(gid<n-1 && gid>0)
    {
        printf("gid: %d \t n: %d\t Li_1[gid]: $lf\tLi[gid]: %lf\tBi[0]: %lf\tz[gid]: %lf\n",gid, size, Li_1[gid], Li[gid], Bi[0],z[gid]);
        Li_1[gid] = xdiff[gid-1] / Li[gid-1];
        printf(" calculated Li_1[gid]\n");
        Li[gid] = sqrt(2*(xdiff[gid-1]+xdiff[gid]) - Li_1[gid-1] * Li_1[gid-1]);
        printf(" calculated Li[gid]\n");
        Bi[0] = 6*(ydiff[gid]/xdiff[gid] - ydiff[gid-1]/xdiff[gid-1]); // Single value
        printf(" calculated Bi\n");
        z[gid] = (Bi[0] - Li_1[gid-1]*z[gid-1])/Li[gid];
        printf(" calculated z[gid]\n");
    }
    for(int i=1; i<n;i++)
    {
        res_Li_1[gid] = xdiff[i-1] / Li[i-1];
        res_Li[gid] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1]);
        res_Bi[0] = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1]); // Single value
        res_z[gid] = (Bi[0] - Li_1[i-1]*z[i-1])/Li[i];
    }
}
*/

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
