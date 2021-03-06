#include <iostream>
#include <exception>
#include <functional>
#include <vector>
#include <cxxopts.hpp>
#include <cutf/device.hpp>
#include "kan.hpp"
#include "gpu_monitor.hpp"
#include "hyperparameter.hpp"

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
std::function<void(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t, std::vector<hyperparameter::parameter_t>)> get_run_function(const std::string type_name){
	if(type_name == "float") return [](int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c, std::vector<hyperparameter::parameter_t> args)
		{kan::run<float>(gpu_id, algorithm_id, string_mode_id, computing_c, args);};
	if(type_name == "double") return [](int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c, std::vector<hyperparameter::parameter_t> args)
		{kan::run<double>(gpu_id, algorithm_id, string_mode_id, computing_c, args);};
	throw std::runtime_error("No such a type : " + type_name);
}
std::function<void(int, kan::algorithm_id, gpu_monitor::string_mode_id, std::size_t)> get_optimize_function(const std::string type_name){
	if(type_name == "float") return [](int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c)
		{kan::optimize<float>(gpu_id, algorithm_id, string_mode_id, computing_c);};
	if(type_name == "double") return [](int gpu_id, kan::algorithm_id algorithm_id, gpu_monitor::string_mode_id string_mode_id, std::size_t computing_c)
		{kan::optimize<double>(gpu_id, algorithm_id, string_mode_id, computing_c);};
	throw std::runtime_error("No such a type : " + type_name);
}
std::vector<hyperparameter::parameter_t> get_hyperparameters_from_string(const std::string str){
	std::vector<hyperparameter::parameter_t> run_arguments;
	std::size_t start_pos = 0;
	std::size_t end_pos = 0;
	while((end_pos = str.find(":", start_pos)) != std::string::npos){
		const auto parameter = std::stol(str.substr(start_pos, (end_pos - start_pos)));
		run_arguments.push_back(parameter);
		start_pos = end_pos + 1;
	}
	// 最後の数字だけは後ろに:がないので別処理
	if(str.length() != 0){
		const auto parameter = std::stol(str.substr(start_pos, (str.length() - start_pos + 1)));
		run_arguments.push_back(parameter);
	}

	return run_arguments;
}
}

int main(int argc, char** argv){
	const std::string project_name = "High Performance ATSUKAN Computing";
	cxxopts::Options options(project_name, "Options");
	options.add_options()
		("a,algorithm", "Computing algorithm", cxxopts::value<std::string>()->default_value("julia"))
		("g,gpu", "GPU ID", cxxopts::value<unsigned int>()->default_value("0"))
		("o,output_mode", "Output mode (csv/human)", cxxopts::value<std::string>()->default_value("human"))
		("s,second", "Heating time[s]", cxxopts::value<std::size_t>()->default_value("5"))
		("t,type", "Computing type", cxxopts::value<std::string>()->default_value("float"))
		("p,parameters", "Hyperparameters (e.g. 1024:2:3)", cxxopts::value<std::string>())
		("opt", "Run optimization")
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

	std::cerr
		<<"# Device information"<<std::endl
		<<"  - GPU ID               : "<<gpu_id<<std::endl
		<<"  - GPU name             : "<<device_prop.name<<std::endl
		<<"  - #SM                  : "<<num_sm<<std::endl
		<<"  - Clock rate           : "<<device_prop.clockRate<<" kHz"<<std::endl;
	std::cerr<<std::endl;
	// }}}
	
	const auto algorithm_name = args["algorithm"].as<std::string>();
	const auto type_name = args["type"].as<std::string>();
	const auto algorithm_id = get_algorithm_id(algorithm_name);
	const auto execution_time = args["second"].as<std::size_t>();
	const auto string_mode_name = args["output_mode"].as<std::string>();
	const auto string_mode_id = get_string_mode_id(string_mode_name);
	
	// print algorithm information {{{
	std::cerr
		<<"# Execution information"<<std::endl
		<<"  - Algorithm name       : "<<algorithm_name<<std::endl
		<<"  - Computing type       : "<<type_name<<std::endl
		<<"  - Execution Time       : "<<execution_time<<" [s]"<<std::endl;
	std::cerr<<std::endl;
	// }}}

	// print output information {{{
	std::cerr
		<<"# Output information"<<std::endl
		<<"  - Output mode          : "<<string_mode_name<<std::endl;
	std::cerr<<std::endl;
	// }

	// run {{{
	if(args.count("opt")){
		const auto optimize_function = get_optimize_function(type_name);
		optimize_function(gpu_id, algorithm_id, string_mode_id, execution_time);
	}else{
		const auto run_function = get_run_function(type_name);
		const auto run_arguments = (args.count("parameters") != 0) ? get_hyperparameters_from_string(args["parameters"].as<std::string>()) : std::vector<hyperparameter::parameter_t>{};
		run_function(gpu_id, algorithm_id, string_mode_id, execution_time, run_arguments);
	}
	// }}}
}
