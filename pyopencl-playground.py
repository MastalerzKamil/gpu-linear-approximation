import pyopencl as cl

if __name__ == "__main__":
    platforms = cl.get_platforms()
    len(platforms)
    gpu_devices = platforms[0].get_devices(cl.device_type.GPU)
    print(gpu_devices)

    cpu_devices = platforms[0].get_devices(cl.device_type.CPU)
    print(cpu_devices)