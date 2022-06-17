#ifndef NETPPOACTOR_H 
#define NETPPOACTOR_H
#include <iostream>
#include <torch/torch.h>
#include "util/types.h"
#include "util/TorchUtil.h"
#include "search_modules/Net/PPO/PPO.h"
using namespace torch;
using std::cout; using std::endl;

// Discrete Actor
struct NetPPOActorImpl: nn::Module
{       
    int64_t state_dim;
    int64_t action_dim;
    int64_t hidden_dim;
    nn::Sequential net{nullptr};    
            
    NetPPOActorImpl(const NetPPOOptions &ops);
    torch::Tensor dist(torch::Tensor s);
    torch::Tensor forward(torch::Tensor s, torch::Tensor a);    
};
TORCH_MODULE(NetPPOActor);
#endif