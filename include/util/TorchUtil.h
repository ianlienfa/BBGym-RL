#ifndef TORCHUTIL_H
#define TORCHUTIL_H

#include <torch/torch.h>

void layer_weight_print(const torch::nn::Module &net);
void print_modules(const torch::nn::Module& module, size_t level = 0);

#define GRAD_TOGGLE(net, on) \
    if (on) { \
        for(auto& p : net->parameters()) \
            p.requires_grad_(true); \
    } else { \
        for(auto& p : net->parameters()) \
                p.requires_grad_(false); \
    }

#endif