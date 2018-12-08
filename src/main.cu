#include <iostream>
#include <cxxopts.hpp>
#include <cutf/device.hpp>

int main(int argc, char** argv){
	cxxopts::Options options("High Performance ATSUKAN Computing", "Options");
	options.add_options()
		("a,algorithm", "Computing algorithm", cxxopts::value<std::string>()->default_value("julia"))
		("g,gpu", "GPU ID", cxxopts::value<unsigned int>()->default_value("0"))
		("h,help", "Help");
	const auto args = options.parse(argc, argv);

	// print USAGE
	if(args.count("help")){
		std::cout<<options.help({""})<<std::endl;
		return 0;
	}

	const auto device_prop = cutf::cuda::device::get_properties_vector();
	const auto num_sm = device_prop[0].multiProcessorCount;
}
