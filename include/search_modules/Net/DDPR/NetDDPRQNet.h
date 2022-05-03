#ifndef NETDDPRQNET_H
#define NETDDPRQNET_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
#include "search_modules/Net/DDPR/DDPR.h"
#include "util/TorchUtil.h"
#include "util/types.h"
using namespace torch;


struct NetDDPRQNetImpl: nn::Module
{
    int64_t state_dim;
    int64_t action_dim;
    Pdd action_range;
    nn::Sequential net{nullptr};

    // rnn
    int64_t rnn_hidden_size;
    int64_t rnn_num_layers;
    nn::RNN rnn{nullptr};
    torch::Tensor hidden_state;
    
    NetDDPRQNetImpl(const NetDDPROptions &ops);
    torch::Tensor forward(torch::Tensor state, torch::Tensor contour_snapshot, torch::Tensor action);
};
TORCH_MODULE(NetDDPRQNet);

#endif