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
	kan_module(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm) : gpu_id(gpu_id), num_sm(num_sm), num_cuda_core_per_sm(num_cuda_core_per_sm){}

	// 燗アルゴリズムの実行
	// C : 計算回数を制御する変数．最適化実行時は値を小さくして評価を行う．
	// parameters : ハイパーパラメータ．最適化ではこれをいじる．
	virtual void run(const int C, std::vector<int> parameters) = 0;
};

// gemm module
template <class T>
class gemm : public kan_module<T>{
public:
	gemm(const int gpu_id);
	void run(const int C, std::vector<int> parameters);
};

// julia module
template <class T>
class julia : public kan_module<T>{
public:
	julia(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm);
	void run(const int C, std::vector<int> parameters);
};
} // kan_module

#endif // __KAN_MODULE_HPP__
