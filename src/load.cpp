#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "search_modules/Net/DDPR/NetDDPR.h"

using std::vector, std::string, std::cout, std::endl;

int main()
{
    NetDDPRImpl net({
        .state_dim = 1,
        .action_dim = 1,
        .action_range = make_pair(-1, 1),
        .q_path = "",
        .pi_path = "",
        .max_num_contour = 10000,
        .rnn_hidden_size = 16,
        .rnn_num_layers = 1
    });
}