#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "kan_kernel.hpp"

namespace{
template <class T>
__global__ void julia_kernel(T* const ptr){
}
template <class T>
void run_julia_kernel(const int num_sm, const int num_cuda_core_per_sm){
	auto dp = cutf::cuda::memory::get_device_unique_ptr<T>(1);

	julia_kernel<T><<<num_sm * 16, num_cuda_core_per_sm * 4>>>(dp.get());
}

// gemm kernel
template <class T>
void run_gemm_kernel(const int num_sm, const int num_cuda_core_per_sm){
}
}

// kernel selector
template <class T>
void kan_kernel::run_kan_kernel(const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algo){

	switch (algo) {
	case kan::julia:
		run_julia_kernel<T>(num_sm, num_cuda_core_per_sm);
		break;
	case kan::gemm:
		run_gemm_kernel<T>(num_sm, num_cuda_core_per_sm);
		break;
	default:
		break;
	}
}

// instance
template void kan_kernel::run_kan_kernel<float>(int, int, kan::algorithm_id);
template void kan_kernel::run_kan_kernel<double>(int, int, kan::algorithm_id);
