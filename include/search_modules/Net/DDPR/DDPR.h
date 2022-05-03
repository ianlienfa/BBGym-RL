#ifndef DDPR_H
#define DDPR_H

#include <iostream>
#include <torch/torch.h>
#include "user_def/oneRjSumCjNode.h"
using namespace torch;

struct NetDDPROptions{
    int64_t state_dim;
    int64_t action_dim;
    Pdd action_range;
    string q_path = "";
    string pi_path = "";
    int64_t max_num_contour = 100;
    int64_t rnn_hidden_size = 16;
    int64_t rnn_num_layers = 1;
};

void initialize_weights_norm(nn::Module& module) {
    torch::NoGradGuard no_grad;
    if (auto* linear = module.as<nn::Linear>()) {
        torch::nn::init::xavier_normal_(linear->weight);
        torch::nn::init::constant_(linear->bias, 0.01);
}

#endif