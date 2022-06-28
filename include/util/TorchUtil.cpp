#include "TorchUtil.h"

#if LAYER_WEIGHT_DEBUG >= 0
void layer_weight_print(const torch::nn::Module &net)
{    
    using namespace std;
    cout << "Printing layer weights of " << net.name() << endl;
    for (auto& p : net.named_parameters()) {

        torch::NoGradGuard no_grad;

        // Access name.
        std::cout << p.key() << std::endl;

        // Access weigth and bias.
        std::cout << p.value() << std::endl;

        // print requires_grad
        std::cout << p.value().requires_grad() << std::endl;

        // print grad_fn
        if(p.value().grad_fn())
          std::cout << p.value().grad_fn()->name() << std::endl;
        else  
          std::cout << "Leaf, No grad_fn" << std::endl;

        // print grad
        std::cout << p.value().grad() << std::endl;
        
        // only print one layer
        break;
    }
};
#else
void layer_weight_print(const torch::nn::Module &net)
{    

};
#endif
void tabs(size_t num) {
  for (size_t i = 0; i < num; i++) {
	std::cout << "\t";
  }
}

void print_modules(const torch::nn::Module& module, size_t level) {
  std::cout << module.name() << " (\n";	
  
  for (const auto& parameter : module.named_parameters()) {
	tabs(level + 1);
	std::cout << parameter.key() << '\t';
	std::cout << parameter.value().sizes() << '\n';
  // std::cout << parameter.value() << '\n';
  } 

  tabs(level);
  std::cout << ")\n";
}

