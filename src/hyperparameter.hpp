#ifndef __HYPERPARAMETER_HPP__
#define __HYPERPARAMETER_HPP__
#include <string>
#include <functional>

namespace hyperparameter{
using parameter_t = long;
struct range{
	std::string name;
	std::string description;
	parameter_t min, max;
	std::function<parameter_t(parameter_t)> get_next;
};
}

#endif // __HYPERPARAMETER_HPP__
