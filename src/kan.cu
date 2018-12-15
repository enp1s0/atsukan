#include <iostream>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include <functional>
#include <memory>
#include <thread>
#include <chrono>
#include "kan.hpp"
#include "kan_algorithm.hpp"

namespace{
template <class T>
std::unique_ptr<kan_algorithm::kan_base<T>> get_kan_algorithm(const int gpu_id, kan::algorithm_id algorithm_id){
	kan_algorithm::kan_base<T>* kan_algorithm_ptr = nullptr;
	switch (algorithm_id) {
	case kan::algorithm_id::gemm:
		kan_algorithm_ptr = new kan_algorithm::gemm<T>(gpu_id);
		break;
	case kan::algorithm_id::julia:
		kan_algorithm_ptr = new kan_algorithm::julia<T>(gpu_id);
		break;
	case kan::algorithm_id::n_body:
		kan_algorithm_ptr = new kan_algorithm::n_body<T>(gpu_id);
		break;
	default:
		; // 世界で一番簡単な文
	}
	return std::unique_ptr<kan_algorithm::kan_base<T>>{kan_algorithm_ptr};
}
template <class T>
double run_core(const int gpu_id, const std::unique_ptr<kan_algorithm::kan_base<T>> &kan_algorithm, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time, const std::vector<hyperparameter::parameter_t>& run_arguments){
	try{
		// real elapsed time
		const auto start_clock = std::chrono::system_clock::now();
		// start kan thread {{{
		bool kan_complete = false;
		std::size_t loop_count;
		std::thread kan_thread([&kan_algorithm, &kan_complete, &run_arguments, &loop_count](){loop_count = kan_algorithm.get()->run(kan_complete, run_arguments);});
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
		bool nvml_not_supported = false;
		for(std::size_t time = 0; time < computing_time; time++){
			const auto elapsed_time = std::time(nullptr) - start_timestamp;
			try{
				monitor.get_gpu_status();
			}catch(std::exception& e){
				if(nvml_not_supported == false){
					std::cerr<<e.what()<<std::endl;
				}
				nvml_not_supported = true;
			}
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
		const auto end_clock = std::chrono::system_clock::now();

		const auto max_power = monitor.get_max_power()/1000.0;
		if(string_mode_id != gpu_monitor::none){
			const auto max_temperature = monitor.get_max_temperature();
			std::cerr<<std::endl;
			std::cerr<<"# Result"<<std::endl
				<<"  - max temperature      : "<<max_temperature<<"C"<<std::endl;
			if(nvml_not_supported){
				std::cerr<<"  - loop count           : "<<loop_count<<std::endl;
			}else{
				std::cerr<<"  - max power            : "<<max_power<<"W"<<std::endl
					<<"  - loop count           : "<<loop_count<<std::endl;
			}
			std::cerr<<"  - real elapsed time    : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - start_clock).count()/1000.0<<"s"<<std::endl;
		}

		return nvml_not_supported ? static_cast<double>(loop_count) : max_power;
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
void print_hyperparameter_informaition(const std::vector<hyperparameter::range>& parameter_ranges){
	// 範囲を表示
	std::cerr<<"# Hyperparameters information"<<std::endl;
	for(const auto& p : parameter_ranges){
		std::cerr<<"  - "<<p.name<<std::endl;
		std::cerr<<"    - description        : "<<p.description<<std::endl;
		std::cerr<<"    - range              : "<<p.min<<" ~ "<<p.max<<std::endl;
	}
	std::cerr<<std::endl;
}
void print_hyperparameter_values(const std::vector<hyperparameter::parameter_t> &params){
	std::cerr<<"  - hyperparameters      : ";
	for(const auto& p : params){
		std::cerr<<p<<",";
	}
	std::cerr<<std::endl;
	std::cerr<<std::endl;
}
} // noname namespace

template <class T>
double kan::run(const int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time, std::vector<hyperparameter::parameter_t> run_arguments){
	const auto kan_algorithm = get_kan_algorithm<T>(gpu_id, algorithm_id);
	const auto parameter_ranges = kan_algorithm.get()->get_hyperparameter_ranges();

	if(string_mode_id != gpu_monitor::string_mode_id::none){
		print_hyperparameter_informaition(parameter_ranges);
		print_hyperparameter_values(run_arguments);
	}

	return run_core<T>(gpu_id, kan_algorithm, string_mode_id, computing_time, run_arguments);
}

template <class T>
void kan::optimize(const int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time){
	const auto kan_algorithm = get_kan_algorithm<T>(gpu_id, algorithm_id);
	const auto parameter_ranges = kan_algorithm.get()->get_hyperparameter_ranges();

	if(string_mode_id != gpu_monitor::string_mode_id::none){
		print_hyperparameter_informaition(parameter_ranges);
	}

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
			print_hyperparameter_values(params);
			std::cout<<"    - value              : "<<power<<std::endl;
		}
		if(power > max_power){
			max_power = power;
			std::copy(params.begin(), params.end(), max_params.begin());
		}
	}while(! update_hyperparameter(params, parameter_ranges));

	std::cerr<<std::endl;
	std::cerr<<"# Optimization result"<<std::endl;
	std::cout<<"  - best parameters      : ";
	for(const auto& p : max_params){
		std::cout<<p<<",";
	}
	std::cout<<std::endl;
	std::cout<<"  - max value            : "<<max_power<<std::endl;

}

template double kan::run<float>(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t, std::vector<hyperparameter::parameter_t>);
template double kan::run<double>(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t, std::vector<hyperparameter::parameter_t>);
template void kan::optimize<float>(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
template void kan::optimize<double>(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
// instance
