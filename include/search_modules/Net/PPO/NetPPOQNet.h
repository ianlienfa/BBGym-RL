#ifndef NETPPOQNET_H
#define NETPPOQNET_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
#include "search_modules/Net/PPO/PPO.h"
#include "util/TorchUtil.h"
#include "util/types.h"
using namespace torch;


struct NetPPOQNetImpl: nn::Module
{
    int64_t state_dim;
    int64_t action_dim;
    int64_t hidden_dim;
    Pdd action_range;
    nn::Sequential net{nullptr};

    NetPPOQNetImpl(const NetPPOOptions &ops);
    torch::Tensor forward(torch::Tensor state);
};
TORCH_MODULE(NetPPOQNet);

#endif