#ifndef NETDDPRACTOR_H 
#define NETDDPRACTOR_H
#include <iostream>
#include <torch/torch.h>
#include "util/types.h"
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
using namespace torch;

struct NetDDPRActorImpl: nn::Module
{   
    Pdd action_range;
    int64_t state_dim;
    nn::Sequential net{nullptr};
    
    NetDDPRActorImpl(int64_t state_dim, Pdd action_range);    
    torch::Tensor forward(torch::Tensor s);
};
TORCH_MODULE(NetDDPRActor);
#endif