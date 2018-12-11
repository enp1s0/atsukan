#include <iostream>
#include <exception>
#include <functional>
#include <cxxopts.hpp>
#include <cutf/device.hpp>
#include <helper_cuda.h>
#include "kan.hpp"
#include "gpu_monitor.hpp"

namespace{
kan::algorithm_id get_algorithm_id(const std::string algorithm_name){
	if(algorithm_name == "julia") return kan::algorithm_id::julia;
	if(algorithm_name == "gemm") return kan::algorithm_id::gemm;
	if(algorithm_name == "n_body") return kan::algorithm_id::n_body;
	throw std::runtime_error("No such an algorithm : " + algorithm_name);
}
gpu_monitor::string_mode_id get_string_mode_id(const std::string string_mode_name){
	if(string_mode_name == "human") return gpu_monitor::string_mode_id::human;
	if(string_mode_name == "csv") return gpu_monitor::string_mode_id::csv;
	throw std::runtime_error("No such printing mode : " + string_mode_name);
}

// 計算型の文字列を受け取ってtemplate引数を設定した関数を返す
std::function<void(int, int, int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t)> get_run_function(const std::string type_name){
	if(type_name == "float") return [](int gpu_id, int num_sm, int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c)
		{kan::run<float>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id, string_mode_id, computing_c);};
	if(type_name == "double") return [](int gpu_id, int num_sm, int num_cuda_core_per_sm, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c)
		{kan::run<double>(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id, string_mode_id, computing_c);};
	throw std::runtime_error("No such a type : " + type_name);
}
}

int main(int argc, char** argv){
	const std::string project_name = "High Performance ATSUKAN Computing";
	cxxopts::Options options(project_name, "Options");
	options.add_options()
		("a,algorithm", "Computing algorithm", cxxopts::value<std::string>()->default_value("julia"))
		("g,gpu", "GPU ID", cxxopts::value<unsigned int>()->default_value("0"))
		("p,print_mode", "Printig mdoe", cxxopts::value<std::string>()->default_value("human"))
		("t,type", "Computing type", cxxopts::value<std::string>()->default_value("float"))
		("h,help", "Help");
	const auto args = options.parse(argc, argv);

	// print USAGE {{{
	if(args.count("help")){
		std::cerr<<options.help({""})<<std::endl;
		return 0;
	}
	// }}}
	std::cerr<<project_name<<std::endl;
	std::cerr<<std::endl;

	// print GPU Information {{{
	const auto gpu_id = args["gpu"].as<unsigned int>();

	const auto device_props = cutf::cuda::device::get_properties_vector();
	if(device_props.size() <= gpu_id){
		throw std::runtime_error("No such a GPU : GPU ID = " + std::to_string(gpu_id));
	}
	const auto device_prop = device_props[gpu_id];
	const int num_sm = device_prop.multiProcessorCount;
	const int num_cuda_core_per_sm = _ConvertSMVer2Cores(device_prop.major, device_prop.minor);

	std::cerr
		<<"# Device information"<<std::endl
		<<"  - GPU ID               : "<<gpu_id<<std::endl
		<<"  - GPU name             : "<<device_prop.name<<std::endl
		<<"  - #SM                  : "<<num_sm<<std::endl
		<<"  - #CUDA Cores per a SM : "<<num_cuda_core_per_sm<<std::endl
		<<"  - Clock rate           : "<<device_prop.clockRate<<" kHz"<<std::endl;
	std::cerr<<std::endl;
	// }}}
	

	// print algorithm information {{{
	const auto algorithm_name = args["algorithm"].as<std::string>();
	const auto type_name = args["type"].as<std::string>();
	const auto algorithm_id = get_algorithm_id(algorithm_name);
	const auto run_function = get_run_function(type_name);
	std::cerr
		<<"# Algorithm information"<<std::endl
		<<"  - Algorithm name       : "<<algorithm_name<<std::endl
		<<"  - Computing type       : "<<type_name<<std::endl;
	std::cerr<<std::endl;
	// }}}
	
	// print output information {{{
	const auto string_mode_name = args["print_mode"].as<std::string>();
	const auto string_mode_id = get_string_mode_id(string_mode_name);
	std::cerr
		<<"# Output information"<<std::endl
		<<"  - Output string type   : "<<string_mode_name<<std::endl;
	std::cerr<<std::endl;

	// }
	
	// run {{{
	run_function(gpu_id, num_sm, num_cuda_core_per_sm, algorithm_id, string_mode_id, 1 << 15);
	// }}}
}
