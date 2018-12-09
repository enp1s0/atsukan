#include "kan_algorithm.hpp"

template <class T>
kan_algorithm::gemm<T>::gemm(const int gpu_id) : kan_algorithm::kan_module<T>(gpu_id, 0, 0){}

template <class T>
void kan_algorithm::gemm<T>::run(const int C, std::vector<int>& parameters){
	// 席を計算する行列の大きさ N x N
	std::size_t N = 1 << 12;

	auto dA = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const T alpha = cutf::cuda::type::cast<T>(0.0f);
	const T beta = cutf::cuda::type::cast<T>(0.0f);

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
