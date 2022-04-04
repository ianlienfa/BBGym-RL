#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(int64_t state_dim, Pdd action_range)
{
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, 64),
            nn::Tanh(),
            nn::Linear(64, 32),
            nn::Tanh(),
            nn::Linear(32, 2)
        )
    );       
    this->state_dim = state_dim;
    this->action_range = action_range;
}

torch::Tensor NetDDPRActorImpl::forward(torch::Tensor s)
{        
    const float &limit = (float) action_range.second;
    torch::Tensor linear_output = net->forward(s);
    auto action_and_prob = linear_output.unbind(1);

    // prob
    torch::Tensor prob = (action_and_prob[1]).sigmoid();

    // action 
    torch::Tensor raw_action = action_and_prob[0];
    torch::Tensor extened_sigmoid_action = 1 / torch::exp(raw_action * (-limit) + 1);
    torch::Tensor output = extened_sigmoid_action * limit;
    output = output.where(prob < 0.4, output.floor());       
    output = output.unsqueeze(1);      
    return output;
}