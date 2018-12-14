#ifndef __KAN_MODULE_HPP__
#define __KAN_MODULE_HPP__
#include <vector>
#include "hyperparameter.hpp"

namespace kan_algorithm{
template <class T>
class kan_base{
private:
	int gpu_id;
	int num_sm;
	int num_cuda_core_per_sm;
public:
	kan_base(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm) : gpu_id(gpu_id), num_sm(num_sm), num_cuda_core_per_sm(num_cuda_core_per_sm){}

	// 燗アルゴリズムの実行
	// c : 計算回数を制御する変数．最適化実行時は値を小さくして評価を行う．
	// parameters : ハイパーパラメータ．最適化ではこれをいじる．
	virtual std::size_t run(const bool& complete, std::vector<hyperparameter::parameter_t> parameters) = 0;
	virtual std::vector<hyperparameter::range> get_hyperparameter_ranges() const = 0;
};

// gemm module
template <class T>
class gemm : public kan_base<T>{
public:
	gemm(const int gpu_id);
	// parameters[0] : 行列サイズ N (N x N)
	std::size_t run(const bool& complete, std::vector<hyperparameter::parameter_t> parameters);
	std::vector<hyperparameter::range> get_hyperparameter_ranges() const;
};

// julia module
template <class T>
class julia : public kan_base<T>{
public:
	julia(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm);
	// parameters[0] : 領域サイズdim (dim x dim)
	// parameters[1] : gridサイズ
	// parameters[2] : blockサイズ
	std::size_t run(const bool& complete, std::vector<hyperparameter::parameter_t> parameters);
	std::vector<hyperparameter::range> get_hyperparameter_ranges() const;
};

// n-body module
template <class T>
class n_body : public kan_base<T>{
public:
	n_body(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm);
	// parameters[0] : 星の数
	// parameters[1] : gridサイズ
	// parameters[2] : blockサイズ
	std::size_t run(const bool& complete, std::vector<hyperparameter::parameter_t> parameters);
	std::vector<hyperparameter::range> get_hyperparameter_ranges() const;
};
} // kan_module

#endif // __KAN_MODULE_HPP__
