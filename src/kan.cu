#include <iostream>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include "kan.hpp"

template <class T>
void kan::run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id){

	// monitoring GPU {{{
	gpu_monitor::monitor gpu_monitor(gpu_id);
	bool kan_complete = false;
	const auto start_timestamp = std::time(nullptr);
	if(string_mode_id == gpu_monitor::csv){
		std::cout<<"elapsed_time,";
	}
	std::cout<<gpu_monitor.get_gpu_status_pre_string(string_mode_id)<<std::endl;
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
}

template void kan::run<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
template void kan::run<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
// instance
