#ifndef __KAN_ALGORITHM_HPP__
#define __KAN_ALGORITHM_HPP__
#include "gpu_monitor.hpp"

namespace kan{
enum algorithm_id{
	julia,
	gemm
};
template <class T>
void run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, const algorithm_id algo_id, const gpu_monitor::string_mode_id string_mode);
}

#endif //__KAN_ALGORITHM_HPP__
