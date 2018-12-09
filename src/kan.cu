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
	kan_algorithm::kan_base<T>* ptr = nullptr;
	switch (algorithm_id) {
	case kan::algorithm_id::gemm:
		ptr = new kan_algorithm::gemm<T>(gpu_id);
	case kan::algorithm_id::julia:
		ptr = new kan_algorithm::julia<T>(gpu_id, num_sm, num_cuda_core_per_sm);
	default:
		; // 世界で一番簡単な文
	}
	return std::unique_ptr<kan_algorithm::kan_base<T>>{ptr};
}
}

template <class T>
void kan::run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id){
	// start kan thread {{{
	bool kan_complete = false;
	auto kan_algorithm = get_kan_algorithm<T>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id);
	std::thread kan_thread([&kan_algorithm](){kan_algorithm.get()->run(3, {1024});});
	// }}}

	// monitoring GPU {{{
	gpu_monitor::monitor gpu_monitor(gpu_id);
	const auto start_timestamp = std::time(nullptr);
	if(string_mode_id == gpu_monitor::csv){
		std::cerr<<"elapsed_time,";
	}
	std::cerr<<gpu_monitor.get_gpu_status_pre_string(string_mode_id)<<std::endl;
	while(!kan_complete){
		const auto elapsed_time = std::time(nullptr) - start_timestamp;
		if(string_mode_id == gpu_monitor::csv){
			std::cout<<elapsed_time<<",";
		}else{
			std::cout<<"["<<std::setw(6)<<elapsed_time<<"] ";
		}
		std::cout<<gpu_monitor.get_gpu_status_string(string_mode_id)<<std::endl;
		sleep(1);
	}
	// }}}
	kan_thread.join();

	std::cerr<<std::endl;
	std::cerr<<"# Result"<<std::endl
		<<"  - max temperature      : "<<gpu_monitor.get_max_temperature()<<"C"<<std::endl
		<<"  - max power            : "<<(gpu_monitor.get_max_power()/1000.0)<<"W"<<std::endl;
}

template void kan::run<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
template void kan::run<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
// instance
