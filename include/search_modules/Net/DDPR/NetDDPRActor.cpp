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
    const int64_t &limit = action_range.second;
    auto netout = net->forward(s);
    #if DEBUG_LEVEL >= 2    
    cout << "netout: " << netout << endl;
    #endif
    return netout * limit;
}