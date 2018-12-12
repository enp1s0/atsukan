#include <type_traits>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include "kan_algorithm.hpp"

namespace{
// convergence_n : 収束とみなす計算回数
template <class T, std::size_t convergence_n = 65536>
__global__ void kernel_julia(T* const output, const std::size_t dim){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= dim * dim) return;

	// 各種定数
	// z = z^z + c
	T z_r = cutf::cuda::type::cast<T>(0.5 - 1.0 * (tid % dim) / dim);
	T z_i = cutf::cuda::type::cast<T>(0.5 - 1.0 * tid / dim);
	T c_r = cutf::cuda::type::cast<T>(-0.3);
	T c_i = cutf::cuda::type::cast<T>(-0.63);

#pragma unroll
	for(auto i = decltype(convergence_n)(0); i < convergence_n; i++){
		z_r = z_r * z_r - z_i * z_i + c_r;
		z_i = cutf::cuda::type::cast<T>(2.0) * z_r * z_i + c_i;
	}
	output[tid] = z_r * z_r + z_i * z_i;
}

}

template <class T>
kan_algorithm::julia<T>::julia(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm) : kan_algorithm::kan_base<T>(gpu_id, num_sm, num_cuda_core_per_sm){
	kan_algorithm::kan_base<T>::arg_ranges.push_back({"dim (field size : dim x dim)", (1<<5), (1<<14), [](const hyperparameter::parameter_t a){return 2 * a;}});
	kan_algorithm::kan_base<T>::arg_ranges.push_back({"threads per a block", (1<<5), (1<<10), [](const hyperparameter::parameter_t a){return 2 * a;}});
}

template <class T>
void kan_algorithm::julia<T>::run(const bool& complete, std::vector<int> parameters){
	const std::size_t dim = parameters[0];
	const std::size_t block_size = parameters[1];

	// memory
	auto d_output = cutf::cuda::memory::get_device_unique_ptr<T>(dim * dim);
	auto h_output = cutf::cuda::memory::get_host_unique_ptr<T>(dim * dim);

	while(!complete){
		kernel_julia<T><<<((dim*dim + block_size - 1)/block_size), block_size>>>(d_output.get(), dim);
		cudaDeviceSynchronize();
	}
}

template class kan_algorithm::julia<float>;
template class kan_algorithm::julia<double>;
