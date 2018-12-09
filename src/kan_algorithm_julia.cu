#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include "kan_algorithm.hpp"

template <class T>
kan_algorithm::julia<T>::julia(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm) : kan_algorithm::kan_base<T>(gpu_id, num_sm, num_cuda_core_per_sm){}

template <class T>
void kan_algorithm::julia<T>::run(const int C, std::vector<int> parameters){
	const std::size_t dim = parameters[0];
}

template class kan_algorithm::julia<float>;
template class kan_algorithm::julia<double>;
