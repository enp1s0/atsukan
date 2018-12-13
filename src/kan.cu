#include <iostream>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include <functional>
#include <memory>
#include <thread>
#include "kan.hpp"
#include "kan_algorithm.hpp"

namespace{
template <class T>
std::unique_ptr<kan_algorithm::kan_base<T>> get_kan_algorithm(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id){
	kan_algorithm::kan_base<T>* kan_algorithm_ptr = nullptr;
	switch (algorithm_id) {
	case kan::algorithm_id::gemm:
		kan_algorithm_ptr = new kan_algorithm::gemm<T>(gpu_id);
		break;
	case kan::algorithm_id::julia:
		kan_algorithm_ptr = new kan_algorithm::julia<T>(gpu_id, num_sm, num_cuda_core_per_sm);
		break;
	case kan::algorithm_id::n_body:
		kan_algorithm_ptr = new kan_algorithm::n_body<T>(gpu_id, num_sm, num_cuda_core_per_sm);
		break;
	default:
		; // 世界で一番簡単な文
	}
	return std::unique_ptr<kan_algorithm::kan_base<T>>{kan_algorithm_ptr};
}
template <class T>
double run_core(const int gpu_id, const std::unique_ptr<kan_algorithm::kan_base<T>> &kan_algorithm, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time, const std::vector<int>& run_arguments){
	try{
		// start kan thread {{{
		bool kan_complete = false;
		std::thread kan_thread([&kan_algorithm, &kan_complete, &run_arguments](){kan_algorithm.get()->run(kan_complete, run_arguments);});
		// }}}

		// monitoring GPU {{{
		gpu_monitor::monitor monitor(gpu_id);
		const auto start_timestamp = std::time(nullptr);
		if(string_mode_id != gpu_monitor::none){
			if(string_mode_id == gpu_monitor::csv){
				std::cerr<<"elapsed_time,";
			}
			std::cerr<<monitor.get_gpu_status_pre_string(string_mode_id)<<std::endl;
		}
		for(std::size_t time = 0; time < computing_time; time++){
			const auto elapsed_time = std::time(nullptr) - start_timestamp;
			monitor.get_gpu_status();
			if(string_mode_id != gpu_monitor::none){
				if(string_mode_id == gpu_monitor::csv){
					std::cout<<elapsed_time<<",";
				}else{
					std::cout<<"["<<std::setw(6)<<elapsed_time<<"] ";
				}
				std::cout<<monitor.get_gpu_status_string(string_mode_id)<<std::endl;
			}
			sleep(1);
		}
		kan_complete = true;
		// }}}
		kan_thread.join();
		const auto max_power = monitor.get_max_power()/1000.0;
		if(string_mode_id != gpu_monitor::none){
			const auto max_temperature = monitor.get_max_temperature();
			std::cerr<<std::endl;
			std::cerr<<"# Result"<<std::endl
				<<"  - max temperature      : "<<max_temperature<<"C"<<std::endl
				<<"  - max power            : "<<max_power<<"W"<<std::endl;
		}
		return max_power;
	}catch(std::exception&){
		return 0.0;
	}
}

// 再帰的にハイパーパラメータの組み合わせを作る
// 返り値 : hyperparameters[i]がranges[i].maxを超えた場合はranges[i].minに戻る．そうしたらtrue
// 根っこのupdate_hyperparameterがfalseを返せば全ハイパーパラメータの探索が行われたことになる
bool update_hyperparameter(std::vector<hyperparameter::parameter_t>& hyperparameters, const std::vector<hyperparameter::range>& ranges, std::size_t index = 0){
	// 一番外側(?)のパラメータでない場合
	if(index < ranges.size() - 1){
		// 1つ外側のパラメータを設定
		const auto changed = update_hyperparameter(hyperparameters, ranges, index + 1);
		// 外側が1順していた場合は次のパラメータにする
		if(changed){
			const auto next = ranges[index].get_next(hyperparameters[index]);
			// maxより大きければminに戻しtrueを返す
			if(next > ranges[index].max){
				hyperparameters[index] = ranges[index].min;
				return true;
			}else{
				hyperparameters[index] = next;
				return false;
			}
		}else{
			return false;
		}
	}else{
		const auto next = ranges[index].get_next(hyperparameters[index]);
		// maxより大きければminに戻しtrueを返す
		if(next > ranges[index].max){
			hyperparameters[index] = ranges[index].min;
			return true;
		}else{
			hyperparameters[index] = next;
			return false;
		}
	}
}
} // noname namespace

template <class T>
double kan::run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time, std::vector<int> run_arguments){
	const auto kan_algorithm = get_kan_algorithm<T>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id);
	return run_core<T>(gpu_id, kan_algorithm, string_mode_id, computing_time, run_arguments);
}

template <class T>
void kan::optimize(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time){
	const auto kan_algorithm = get_kan_algorithm<T>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id);
	const auto parameter_ranges = kan_algorithm.get()->get_hyperparameter_ranges();

	std::vector<hyperparameter::parameter_t> max_params(parameter_ranges.size());
	double max_power = 0.0;
	// まず，ハイパーパラメータにそれぞれの最小値をセット
	std::vector<hyperparameter::parameter_t> params;
	for(const auto p : parameter_ranges){
		params.push_back(p.min);
	}
	std::size_t experiment_id = 0;
	do{
		// 燗関数を実行
		const auto power = run_core<T>(gpu_id, kan_algorithm, gpu_monitor::string_mode_id::none, computing_time, params);
		// CSV形式で結果を表示
		if(string_mode_id == gpu_monitor::string_mode_id::csv){
			for(const auto& p : params){
				std::cout<<p<<",";
			}
			std::cout<<power<<std::endl;
		}
		// 人間に優しく表示
		else if(string_mode_id == gpu_monitor::string_mode_id::human){
			std::cout<<"## Experiment "<<(experiment_id++)<<std::endl;
			std::cout<<"    - parameters         : ";
			for(const auto& p : params){
				std::cout<<p<<",";
			}
			std::cout<<std::endl;
			std::cout<<"    - power              : "<<power<<"W"<<std::endl;
		}
		if(power > max_power){
			max_power = power;
			std::copy(params.begin(), params.end(), max_params.begin());
		}
	}while(! update_hyperparameter(params, parameter_ranges));

	std::cerr<<"# Optimization result"<<std::endl;
	std::cout<<"  - best parameters      : ";
	for(const auto& p : max_params){
		std::cout<<p<<",";
	}
	std::cout<<std::endl;
	std::cout<<"  - max power            : "<<max_power<<"W"<<std::endl;

}

template double kan::run<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t, std::vector<int>);
template double kan::run<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t, std::vector<int>);
template void kan::optimize<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
template void kan::optimize<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
// instance
