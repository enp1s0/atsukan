#include "kan.hpp"

template <class T>
void kan::run(int num_sm, int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id){

}

template void kan::run<float>(int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
template void kan::run<double>(int, int, kan::algorithm_id, gpu_monitor::string_mode_id);
// instance
