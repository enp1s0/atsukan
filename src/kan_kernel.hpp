#ifndef __KAN_KERNEL_HPP__
#define __KAN_KERNEL_HPP__
#include "kan.hpp"

namespace kan_kernel{
template <class T>
void run_kan_kernel(const int num_sm, const int num_cuda_core_per_sm, kan::algorithm_id algo);
}

#endif // __KAN_KERNEL_HPP__
