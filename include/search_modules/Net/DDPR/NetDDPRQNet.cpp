#include "search_modules/Net/DDPR/NetDDPRQNet.h"

NetDDPRQNetImpl::NetDDPRQNetImpl(int64_t state_dim, int64_t action_dim)
{
    this->action_dim = action_dim;
    this->state_dim = state_dim;
    net = register_module("Sequential", nn::Sequential(
        nn::Linear(state_dim+action_dim, 32),
        nn::ReLU(),
        nn::Linear(32, 64),
        nn::ReLU(),
        nn::Linear(64, 32),
        nn::ReLU(),
        nn::Linear(32, 1)
    ));
}

torch::Tensor NetDDPRQNetImpl::forward(torch::Tensor state, torch::Tensor action)
{
    std::cout << "Q(s, a) -- s: " << state.sizes() << " a: " << action.sizes() << std::endl;
    auto input = torch::cat({state, action}, -1);
    if(input.sizes()[0] > 1)
        std::cout << "processing batch!" << std::endl;    
    std::cout << "Q(s, a) -- input: " << input.sizes() << std::endl;
    #if DEBUG_LEVEL >= 0
        std::cout << "input: " << input << std::endl;
    #endif
    auto output_raw = net->forward(input);
    std::cout << "Q(s, a) -- output_raw: " << "size: " << output_raw.sizes() << output_raw << std::endl;
    auto output = torch::squeeze(output_raw, -1);
    std::cout << "Q(s, a) output tensor: " << "size: " << output_raw.sizes() << output << std::endl;
    return output;
}