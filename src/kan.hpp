#ifndef __KAN_ALGORITHM_HPP__
#define __KAN_ALGORITHM_HPP__

namespace kan{
enum algorithm_id{
	julia
};
template <class T>
void run(const int num_sm, const int num_cuda_core_per_sm, const algorithm_id algo_id);
}

#endif //__KAN_ALGORITHM_HPP__
