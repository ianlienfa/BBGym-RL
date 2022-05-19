#ifndef PPO_H
#define PPO_H

#include <iostream>
#include <torch/torch.h>
#include "user_def/oneRjSumCjNode.h"
using namespace torch;

struct NetPPOOptions{
    int64_t state_dim;
    int64_t action_dim;
    int64_t hidden_dim;
    Pdd action_range;
    string q_path = "";
    string pi_path = "";
    int64_t max_num_contour = 100;
};

#endif