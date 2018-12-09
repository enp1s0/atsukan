#ifndef __KAN_MODULE_HPP__
#define __KAN_MODULE_HPP__
#include <vector>

namespace kan_algorithm{
template <class T>
class kan_module{
protected:
	int gpu_id;
	int num_sm;
	int num_cuda_core_per_sm;
public:
	module(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm);

	// 燗アルゴリズムの実行
	virtual void run(std::vector<int>& parameters) = 0;
};
} // kan_module

#endif // __KAN_MODULE_HPP__
