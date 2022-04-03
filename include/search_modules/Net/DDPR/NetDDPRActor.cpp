#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(int64_t state_dim, Pdd action_range)
{
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, 64),
            nn::Tanh(),
            nn::Linear(64, 32),
            nn::Tanh(),
            nn::Linear(32, 2),
            nn::Sigmoid()
        )
    );       
    
    this->state_dim = state_dim;
    this->action_range = action_range;
}

torch::Tensor NetDDPRActorImpl::forward(torch::Tensor s)
{    
    const int64_t &limit = action_range.second;
    torch::Tensor output = net->forward(s);
    auto action_and_prob = output.unbind(1);
    output = action_and_prob[0] * limit;
    torch::Tensor prob = action_and_prob[1];
    output = output.where(prob < 0.2, output.floor());       
    output = output.unsqueeze(1);    
    return output;
}