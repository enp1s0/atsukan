#ifndef __GPU_MONITOR_HPP__
#define __GPU_MONITOR_HPP__
#include <string>
namespace gpu_monitor{
enum string_mode_id{
	human,
	csv
};
std::string get_gpu_status_string();
}

#endif // __GPU_MONITOR_HPP__
