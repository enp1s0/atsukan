#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include "kan_algorithm.hpp"

#if __CUDA_ARCH__ <= 400
#define __ldg(x) (*(x))
#endif

namespace{
template <class T>
__global__ void n_body_compute_velosity_kernel(
			T* const px,
			T* const py,
			T* const pz,
			T* const vx,
			T* const vy,
			T* const vz,
			T* m,
			const T dt,
			const std::size_t N
		){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	constexpr T G = static_cast<T>(6.67408e-11);
	auto reg_fx = static_cast<T>(0);
	auto reg_fy = static_cast<T>(0);
	auto reg_fz = static_cast<T>(0);
	const auto reg_px = __ldg(px + tid);
	const auto reg_py = __ldg(py + tid);
	const auto reg_pz = __ldg(pz + tid);
	const auto reg_m = __ldg(m + tid);
	
	for(std::size_t i = 0; i < N; i++){
		if(i == N) continue;
		const auto rx = __ldg(px + i) - reg_px;
		const auto ry = __ldg(py + i) - reg_py;
		const auto rz = __ldg(pz + i) - reg_pz;
		const auto r2 = rx * rx + ry * ry + rz * rz;
		const auto r = static_cast<T>(sqrt(r2));
		const auto f = G * __ldg(m + i) * reg_m / (r2 * r);
		reg_fx += f * rx;
		reg_fy += f * ry;
		reg_fz += f * rz;
	}
	vx[tid] += reg_fx / reg_m * dt;
	vy[tid] += reg_fy / reg_m * dt;
	vz[tid] += reg_fz / reg_m * dt;
}
template <class T>
__global__ void n_body_compute_position_kernel(
			T* const px,
			T* const py,
			T* const pz,
			T* const vx,
			T* const vy,
			T* const vz,
			const T dt,
			const std::size_t N
		){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	px[tid] += __ldg(vx + tid) * dt;
	py[tid] += __ldg(vy + tid) * dt;
	pz[tid] += __ldg(vz + tid) * dt;
}
}

template <class T>
kan_algorithm::n_body<T>::n_body(const int gpu_id, const int num_sm, const int num_cuda_core_per_sm) : kan_algorithm::kan_base<T>(gpu_id, num_sm, num_cuda_core_per_sm){}

template <class T>
void kan_algorithm::n_body<T>::run(const bool& complete, std::vector<int> parameters){
	const std::size_t N = parameters[0];
	const std::size_t block_size = parameters[1];
	const T dt = static_cast<T>(0.001);

	auto d_px = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_py = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_pz = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_vx = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_vy = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_vz = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto d_m = cutf::cuda::memory::get_device_unique_ptr<T>(N);

	while(!complete){
		n_body_compute_velosity_kernel<T><<<(N + block_size - 1)/block_size, block_size>>>(
				d_px.get(),
				d_py.get(),
				d_pz.get(),
				d_vx.get(),
				d_vy.get(),
				d_vz.get(),
				d_m.get(),
				dt,
				N
				);
		n_body_compute_position_kernel<T><<<(N + block_size - 1)/block_size, block_size>>>(
				d_px.get(),
				d_py.get(),
				d_pz.get(),
				d_vx.get(),
				d_vy.get(),
				d_vz.get(),
				dt,
				N
				);
		cudaDeviceSynchronize();
	}
	cutf::cuda::error::check(cudaGetLastError(), __FILE__, __LINE__, __func__);
}

template class kan_algorithm::n_body<float>;
template class kan_algorithm::n_body<double>;
