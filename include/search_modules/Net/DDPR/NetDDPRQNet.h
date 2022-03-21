#ifndef NETDDPRQNET_H
#define NETDDPRQNET_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
using namespace torch;
using namespace std;


struct NetDDPRQNetImpl: nn::Module
{
    int64_t state_dim;
    int64_t action_dim;
    nn::Sequential net{nullptr};

    NetDDPRQNetImpl(int64_t state_dim, int64_t action_dim);
    torch::Tensor forward(torch::Tensor state, torch::Tensor action);
};
TORCH_MODULE(NetDDPRQNet);

#endif