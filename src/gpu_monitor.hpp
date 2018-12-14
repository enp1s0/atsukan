#ifndef __GPU_MONITOR_HPP__
#define __GPU_MONITOR_HPP__
#include <string>
#include <nvml.h>

namespace gpu_monitor{
enum string_mode_id{
	human,
	csv,
	none
};
class monitor{
	nvmlDevice_t device;
	unsigned int power_max_limit;

	// max value
	unsigned int max_power;
	unsigned int max_temperature;

	// current value
	unsigned int current_temperature;
	unsigned int current_power;
	nvmlPstates_t current_states;
public:
	monitor(unsigned int gpu_id);
	~monitor();
	std::string get_gpu_status_string(const string_mode_id string_mode);
	// CSVの項目名などの情報を表示
	std::string get_gpu_status_pre_string(const string_mode_id string_mode);

	void get_gpu_status();

	// max value getter
	unsigned int get_max_temperature() const;
	unsigned int get_max_power() const;
};
}

#endif // __GPU_MONITOR_HPP__
