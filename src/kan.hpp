#ifndef __KAN_ALGORITHM_HPP__
#define __KAN_ALGORITHM_HPP__
#include <vector>
#include "gpu_monitor.hpp"
#include "hyperparameter.hpp"

namespace kan{
enum algorithm_id{
	julia,
	gemm,
	n_body
};
// 返り値 : max power
template <class T>
double run(const int gpu_id, const algorithm_id algo_id, const gpu_monitor::string_mode_id string_mode, const std::size_t compute_time, std::vector<hyperparameter::parameter_t> arguments);
// 最適化
template <class T>
void optimize(const int gpu_id, const algorithm_id algo_id, const gpu_monitor::string_mode_id string_mode, const std::size_t compute_time);
}

#endif //__KAN_ALGORITHM_HPP__
