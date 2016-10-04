#ifndef _OpenClUtils_h_
#define _OpenClUtils_h_

#include <string>
#include <CL/cl.hpp>

namespace OpenClUtils {

	void checkClError(cl_int err, const char * msg);

	cl::Platform const getDefaultPlatform();
	cl::Device const getDefaultDevice();
	cl::Context const createDefaultDeviceContext();
		
	cl::Program const createProgram(cl::Context context, std::vector<cl::Device> const &devices, std::string const &sourcePath);
	cl::Kernel const createKernel(cl::Program const &program, std::string const &kernelName);

	void printDeviceInfo(cl::Device device);
}


#endif