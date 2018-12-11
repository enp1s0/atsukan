#include <string>
#include <iomanip>
#include <sstream>
#include <cutf/nvml.hpp>
#include "gpu_monitor.hpp"


#define NVML_ERROR_HANDLE(status) cutf::nvml::error::check(status, __FILE__, __LINE__, __func__)

gpu_monitor::monitor::monitor(unsigned int gpu_id) : max_power(0), max_temperature(0){
	NVML_ERROR_HANDLE(nvmlInit());
	NVML_ERROR_HANDLE(nvmlDeviceGetHandleByIndex(gpu_id, &device));
	NVML_ERROR_HANDLE(nvmlDeviceGetEnforcedPowerLimit(device, &power_max_limit));
}
gpu_monitor::monitor::~monitor(){
	NVML_ERROR_HANDLE(nvmlShutdown());
}

// pre print string
// e.g. csv columun title
std::string gpu_monitor::monitor::get_gpu_status_pre_string(const gpu_monitor::string_mode_id string_mode){
	std::string pre_status_string;
	switch (string_mode) {
	case gpu_monitor::human:
		pre_status_string = "Power limit    : " + std::to_string(power_max_limit/1000.0);
		break;
	case gpu_monitor::csv:
		pre_status_string = "temperature,power,power_max_limit";
		break;
	default:
		break;
	}
	return pre_status_string;
}

void gpu_monitor::monitor::get_gpu_status(){
	// get gpu temperature/power {{{
	NVML_ERROR_HANDLE(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &current_temperature));
	NVML_ERROR_HANDLE(nvmlDeviceGetPowerUsage(device, &current_power));
	NVML_ERROR_HANDLE(nvmlDeviceGetPerformanceState(device, &current_states));
	// }}}

	// record max {{{
	max_power = std::max(max_power, current_power);
	max_temperature = std::max(max_temperature, current_temperature);
	// }}}
}

std::string gpu_monitor::monitor::get_gpu_status_string(const gpu_monitor::string_mode_id string_mode){

	std::string status_string = "";
	if(string_mode == gpu_monitor::human){
		std::stringstream ss;
		ss<<"Temp:"<<std::setw(3)<<current_temperature<<"C, Pow:"<<std::setw(5)<<(current_power/1000.0)<<"W, Perf :P" + std::to_string((int) current_states);
		status_string = ss.str();
	}else if(string_mode == gpu_monitor::csv){
		status_string = std::to_string(current_temperature) + "," + std::to_string(current_power/1000.0) + "," + std::to_string(power_max_limit/1000.0);
	}

	return status_string;
}

unsigned int gpu_monitor::monitor::get_max_power() const{
	return max_power;
}

unsigned int gpu_monitor::monitor::get_max_temperature() const{
	return max_temperature;
}
