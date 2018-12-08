#ifndef __GPU_MONITOR_HPP__
#define __GPU_MONITOR_HPP__
#include <string>
#include <nvml.h>

namespace gpu_monitor{
enum string_mode_id{
	human,
	csv
};
class monitor{
	nvmlDevice_t device;
	unsigned int power_max_limit;
public:
	monitor(unsigned int gpu_id);
	~monitor();
	std::string get_gpu_status_string(const string_mode_id string_mode);
	std::string get_gpu_status_pre_string(const string_mode_id string_mode);
};
}

#endif // __GPU_MONITOR_HPP__
