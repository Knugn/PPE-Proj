#ifdef _DEBUG
#define __CL_ENABLE_EXCEPTIONS
#endif

#include <iostream>
#include <fstream>
#include <streambuf>
#include <iterator>

#include "OpenClUtils.h"

namespace OpenClUtils {

	cl::Platform const getDefaultPlatform() {
		std::vector<cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		if(all_platforms.size()==0){
			throw "No platforms found. Check OpenCL installation!";
		}
		cl::Platform default_platform=all_platforms[0];
		return default_platform;
	}

	cl::Device const getDefaultDevice() {
		cl::Platform default_platform = getDefaultPlatform();
		std::vector<cl::Device> all_devices;
		default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		if(all_devices.size()==0){
			throw "No devices found. Check OpenCL installation!";
		}
		cl::Device default_device=all_devices[0];
		return default_device;
	}

	cl::Context const createDefaultDeviceContext() {
		cl::Device default_device = getDefaultDevice();
		cl::Context context(default_device);
		return context;
	}

	/*
	cl::Program const createProgram(cl::Context context, std::vector<cl::Device> const &devices, std::string const &sourcePath) {
		std::ifstream sourceFile(sourcePath);
		if (!sourceFile.is_open())
			std::cerr << "Failed to open kernel source file \"" << sourcePath << "\"" << std::endl;
		std::string kernelSource((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.length()));
		cl::Program program(context, sources);
		cl_int err = program.build(devices);
		for (cl::Device device : devices) {
			OpenClUtils::checkErr(err, 
				("Failed to build program. Build log: \n" + 
				program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)).c_str());
		}
		return program;
	}

	cl::Kernel const createKernel(cl::Program const &program, std::string const &kernelName) {
		cl_int err;
		cl::Kernel kernel(program, kernelName.c_str(), &err);
		OpenClUtils::checkErr(err, ("Failed to locate kernel entry function \"" + kernelName + "\"").c_str());
		return kernel;
	}

	void printDeviceInfo(cl::Device device) {
		cl_int err;
		std::cout << "Listing device info ..." << std::endl;

		auto name = device.getInfo<CL_DEVICE_NAME>(&err);
		checkErr(err, "Failed to get device name.");
		std::cout << " Device name: " << name << std::endl;

		auto type = device.getInfo<CL_DEVICE_TYPE>(&err);
		checkErr(err, "Failed to get device type.");
		std::string type_human_readable;
		if (type == CL_DEVICE_TYPE_GPU)
			type_human_readable = "GPU";
		else if (type == CL_DEVICE_TYPE_CPU)
			type_human_readable = "CPU";
		else
			type_human_readable = "? (Not CPU or GPU)";
		std::cout << " Device type: " << type_human_readable << std::endl;

		auto device_version = device.getInfo<CL_DEVICE_VERSION>(&err);
		checkErr(err, "Failed to get device version.");
		std::cout << " Device version: " << device_version << std::endl;

		auto opencl_c_version = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>(&err);
		checkErr(err, "Failed to get device OpenCL C version.");
		std::cout << " Device OpenCL C version: " << opencl_c_version << std::endl;

		auto driver_version = device.getInfo<CL_DRIVER_VERSION>(&err);
		checkErr(err, "Failed to get driver version.");
		std::cout << " Driver version: " << driver_version << std::endl;

		auto max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
		checkErr(err, "Failed to get max compute units.");
		std::cout << " Max compute units: " << max_compute_units << std::endl;

		auto max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
		checkErr(err, "Failed to get max work group size.");
		std::cout << " Max work group size: " << max_work_group_size << std::endl;

		auto global_mem_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);
		checkErr(err, "Failed to get global memory size.");
		std::cout << " Global memory size: " << global_mem_size << " bytes" << std::endl;

		auto global_mem_cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>(&err);
		checkErr(err, "Failed to get global memory cache size.");
		std::cout << " Global memory cache size: " << global_mem_cache_size << " bytes" << std::endl;

		auto global_mem_cache_line_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>(&err);
		checkErr(err, "Failed to get global memory cache line size.");
		std::cout << " Global memory cacheline size: " << global_mem_cache_line_size << " bytes" << std::endl;

		auto local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);
		checkErr(err, "Failed to get local memory size.");
		std::cout << " Local memory size: " << local_mem_size << " bytes" << std::endl;

		auto preferred_int_vector_width = device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>(&err);
		checkErr(err, "Failed to get preferred int vector width.");
		std::cout << " Preferred int vector width: " << preferred_int_vector_width << " element(s)" << std::endl;

		auto preferred_float_vector_width = device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>(&err);
		checkErr(err, "Failed to get preferred float vector width.");
		std::cout << " Preferred float vector width: " << preferred_float_vector_width << " element(s)" << std::endl;
	}
	*/
	void checkClError(cl_int err, const char * msg) {
		if (err != CL_SUCCESS) {
			std::cerr << "OpenCL ERROR ("<<err<<"): " << msg  << std::endl;
			std::cout << "Press any key to exit." << std::endl;
			std::getchar();
			exit(EXIT_FAILURE);
		}
	}

};