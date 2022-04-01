#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(int64_t state_dim, Pdd action_range)
{
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, 64),
            nn::Tanh(),
            nn::Linear(64, 1),
            nn::Sigmoid()
        )
    );       
    this->state_dim = state_dim;
    this->action_range = action_range;
}

torch::Tensor NetDDPRActorImpl::forward(torch::Tensor s)
{    
    // cout << "NetDDPRActorImpl::forward tensor: " << s << endl;
    const int64_t &limit = action_range.second;
    auto netout = net->forward(s);
    // cout << "pi raw out: " << netout << endl;
    auto output = netout * limit;
    // std::cout << "action output: " << std::endl << output << std::endl;
    return output;
}