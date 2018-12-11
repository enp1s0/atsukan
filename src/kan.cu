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
}

template <class T>
double kan::run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, const std::size_t computing_time){
	// start kan thread {{{
	bool kan_complete = false;
	auto kan_algorithm = get_kan_algorithm<T>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id);
	// 関数を抜けたら完了フラグを立てる
	std::thread kan_thread([&kan_algorithm, &kan_complete](){kan_algorithm.get()->run(kan_complete, {1<<13, 512}); std::cout<<"done"<<std::endl;});
	// }}}

	// monitoring GPU {{{
	gpu_monitor::monitor gpu_monitor(gpu_id);
	const auto start_timestamp = std::time(nullptr);
	if(string_mode_id != gpu_monitor::none){
		if(string_mode_id == gpu_monitor::csv){
			std::cerr<<"elapsed_time,";
		}
		std::cerr<<gpu_monitor.get_gpu_status_pre_string(string_mode_id)<<std::endl;
	}
	for(std::size_t time = 0; time < computing_time; time++){
		const auto elapsed_time = std::time(nullptr) - start_timestamp;
		gpu_monitor.get_gpu_status();
		if(string_mode_id != gpu_monitor::none){
			if(string_mode_id == gpu_monitor::csv){
				std::cout<<elapsed_time<<",";
			}else{
				std::cout<<"["<<std::setw(6)<<elapsed_time<<"] ";
			}
			std::cout<<gpu_monitor.get_gpu_status_string(string_mode_id)<<std::endl;
			sleep(1);
		}
	}
	kan_complete = true;
	// }}}
	kan_thread.join();

	const auto max_power = gpu_monitor.get_max_power()/1000.0;
	if(string_mode_id != gpu_monitor::none){
		const auto max_temperature = gpu_monitor.get_max_temperature();
		std::cerr<<std::endl;
		std::cerr<<"# Result"<<std::endl
			<<"  - max temperature      : "<<max_temperature<<"C"<<std::endl
			<<"  - max power            : "<<max_power<<"W"<<std::endl;
	}

	return max_power;
}

template double kan::run<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
template double kan::run<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t);
// instance
