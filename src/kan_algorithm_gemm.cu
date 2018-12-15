#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "kan_algorithm.hpp"

template <class T>
kan_algorithm::gemm<T>::gemm(const int gpu_id) : kan_algorithm::kan_base<T>(gpu_id){}

template <class T>
std::size_t kan_algorithm::gemm<T>::run(const bool &complete, std::vector<hyperparameter::parameter_t> parameters){
	std::size_t loop_count = 0;
	// 席を計算する行列の大きさ N x N
	const std::size_t N = parameters[0];

	auto dA = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const T alpha = cutf::cuda::type::cast<T>(0.0f);
	const T beta = cutf::cuda::type::cast<T>(0.0f);

	while(!complete){
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
		cudaDeviceSynchronize();
		loop_count++;
	}
	return loop_count;
}

template <class T>
std::vector<hyperparameter::range> kan_algorithm::gemm<T>::get_hyperparameter_ranges() const{
	return {
		{"N","matrix size N x N", 1<<6, 1<<11, [](hyperparameter::parameter_t a){return a * 2;}}
	};
}

template class kan_algorithm::gemm<float>;
template class kan_algorithm::gemm<double>;
