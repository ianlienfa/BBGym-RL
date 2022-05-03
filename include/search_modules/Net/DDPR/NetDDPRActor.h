#ifndef NETDDPRACTOR_H 
#define NETDDPRACTOR_H
#include <iostream>
#include <torch/torch.h>
#include "util/types.h"
#include "search_modules/Net/DDPR/DDPR.h"
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
using namespace torch;
using std::cout; using std::endl;


struct NetDDPRActorImpl: nn::Module
{   
    Pdd action_range;
    int64_t state_dim;
    int64_t split_map[3];
    nn::Sequential net{nullptr};
    vector<float> arg_softmax_map_arr;
    torch::Tensor arg_softmax_map;
            
    NetDDPRActorImpl(const NetDDPROptions &ops);    
    torch::Tensor forward(torch::Tensor s, torch::Tensor contour_snapshot);
};
TORCH_MODULE(NetDDPRActor);
#endif