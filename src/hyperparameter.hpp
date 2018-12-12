#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <string>
#include <functional>

namespace hyperparameter{
using parameter_t = int;
struct range{
	std::string name;
	parameter_t min, max;
	std::function<parameter_t(parameter_t)> get_next;
};
}

#endif // __HYPERPARAMETER_HPP__
