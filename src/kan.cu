#include "kan.hpp"

template <class T>
void kan::run(int num_sm, int num_cuda_core_per_sm, kan::algorithm_id algo){

}

template void kan::run<float>(int, int, kan::algorithm_id);
template void kan::run<double>(int, int, kan::algorithm_id);
// instance
