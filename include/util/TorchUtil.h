#ifndef TORCHUTIL_H
#define TORCHUTIL_H

#include <torch/torch.h>

void layer_weight_print(const torch::nn::Module &net);

// void module_info(const torch::nn::Module &net){
//     for (const auto& p : net.named_parameters()) {
//     std::cout << "=================" << p.key() << "=================" << "\n" << p.value() << std::endl;
//     }
// }


// void tabs(size_t num) {
//   for (size_t i = 0; i < num; i++) {
// 	std::cout << "\t";
//   }
// }

// void print_modules(const torch::nn::Module& module, size_t level = 0) {
//     // std::cout << module.name().qualifiedName() << " (\n";
//     std::cout << module.name().name() << " (\n";	

//     for (const auto& parameter : module.get_parameters()) {
//     tabs(level + 1);
//     std::cout << parameter.name() << '\t';
//     std::cout << parameter.value().toTensor().sizes() << '\n';
//     } 

//     for (const auto& module : module.get_modules()) {
//     tabs(level + 1);
//     print_modules(module, level + 1);
//     }

//     tabs(level);
//     std::cout << ")\n";
// }

#endif