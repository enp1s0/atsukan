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
	// 席を計算する行列の大きさ N x N
	constexpr std::size_t N = 1 << 12;
	// 計算回数
	constexpr std::size_t C = 1 << 14;

	auto dA = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	T alpha = cutf::cuda::type::cast<T>(0.0f);
	T beta = cutf::cuda::type::cast<T>(0.0f);

	for(auto c = decltype(C)(0); c < C; c++){
		cutf::cublas::error::check(cutf::cublas::gemm(
				*cublas.get(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				N, N, N,
				&alpha,
				dA.get(), N,
				dB.get(), N,
				&beta,
				dC.get(), N
				), __FILE__, __LINE__, __func__);
	}
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
