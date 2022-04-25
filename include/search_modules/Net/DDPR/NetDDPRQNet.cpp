#include "search_modules/Net/DDPR/NetDDPRQNet.h"

NetDDPRQNetImpl::NetDDPRQNetImpl(int64_t state_dim, int64_t action_dim, Pdd action_range)
{
    this->action_dim = action_dim;
    this->state_dim = state_dim;
    this->action_range = action_range;
    net = register_module("Sequential", nn::Sequential(
        nn::Linear(state_dim+action_dim, 32),
        nn::ReLU(),
        nn::Linear(32, 64),
        nn::ReLU(),
        // nn::Linear(64, 256),
        // nn::ReLU(),
        // nn::Linear(256, 64),
        // nn::ReLU(),
        nn::Linear(64, 32),
        nn::ReLU(),
        nn::Linear(32, 1)
    ));
}


torch::Tensor NetDDPRQNetImpl::forward(torch::Tensor state, torch::Tensor action)
{
    // test if normalization have effect    
    // action = (action - (action_range.second/2)) / action_range.second; 
    auto input = torch::cat({state, action}, -1);
    #if TORCH_DEBUG >= 1
        std::cout << "action norm: " << action << std::endl;
        if(input.sizes()[0] > 1)
            std::cout << "processing batch!" << std::endl;    
        std::cout << "Q(s, a) -- input: " << input.sizes() << std::endl;
        std::cout << "input: " << input << std::endl;
    #endif
    auto output = net->forward(input);
    if(output.index({0}).item<float>() > 1e10)
    {
        std::cout << "state: " << state << std::endl;
        std::cout << "action: " << action << std::endl;
        throw("Q(s, a) is too large!");
    }
    #if TORCH_DEBUG >= 1
    std::cout << "input: " << input << std::endl;
    std::cout << "Q(s, a): " << output << std::endl;
    std::cout << "output q value: " << output << std::endl;
    #endif
    return output;
}