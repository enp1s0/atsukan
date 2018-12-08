#include <iostream>
#include <ctime>
#include <unistd.h>
#include <iomanip>
#include <functional>
#include <thread>
#include "kan.hpp"
#include "kan_kernel.hpp"

namespace{
template <class T>
std::function<void(void)> get_kan_function(const int num_sm, const int num_cuda_core_per_sm, const kan::algorithm_id algo, bool& kan_complete){
	return [num_sm, num_cuda_core_per_sm, &algo, &kan_complete](){kan_kernel::run_kan_kernel<T>(num_sm, num_cuda_core_per_sm, algo);kan_complete = true;};
}
}

template <class T>
void kan::run(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id){
	// start kan thread {{{
	bool kan_complete = false;
	const auto kan_function = get_kan_function<T>(num_sm, num_cuda_core_per_sm, algorithm_id, kan_complete);
	std::thread kan_thread(kan_function);
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
}

template void kan::run<float>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
template void kan::run<double>(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
// instance
