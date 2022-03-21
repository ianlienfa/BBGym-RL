#ifndef TORCHUTIL_H
#define TORCHUTIL_H

#include <torch/torch.h>
void module_info(const torch::nn::Module &net){
    for (const auto& p : net.named_parameters()) {
    std::cout << "=================" << p.key() << "=================" << "\n" << p.value() << std::endl;
    }
}

#endif