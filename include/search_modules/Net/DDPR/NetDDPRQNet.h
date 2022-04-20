#ifndef NETDDPRQNET_H
#define NETDDPRQNET_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
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
    
    NetDDPRQNetImpl(int64_t state_dim, int64_t action_dim, Pdd action_range, int64_t num_max_contour, int64_t rnn_hidden_size = 1, int64_t rnn_num_layers = 1);
    torch::Tensor forward(torch::Tensor state, torch::Tensor action, torch::Tensor contour_snapshot);
};
TORCH_MODULE(NetDDPRQNet);

#endif