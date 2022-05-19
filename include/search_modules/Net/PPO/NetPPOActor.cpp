#include "search_modules/Net/PPO/NetPPOActor.h"

NetPPOActorImpl::NetPPOActorImpl(const NetPPOOptions& ops)
{    
    this->state_dim = ops.state_dim;
    this->action_dim = ops.action_dim;
    this->hidden_dim = ops.hidden_dim;
    
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, hidden_dim),
            nn::ReLU(),
            nn::Linear(hidden_dim, hidden_dim),
            nn::ReLU(),
            nn::Linear(hidden_dim, action_dim),
            nn::Softmax(nn::SoftmaxOptions(-1 /* dim */))
        )
    );       

    
    {
        torch::NoGradGuard no_grad;
        // weight init
        auto initialize_weights_norm = [](nn::Module& module) {
            torch::NoGradGuard no_grad;
            if (auto* linear = module.as<nn::Linear>()) {
                torch::nn::init::kaiming_uniform_(linear->weight);
                torch::nn::init::constant_(linear->bias, 0.01);
            }
        };
        this->apply(initialize_weights_norm);
    }
}


// Pass back only Q(s, a) for now
torch::Tensor NetPPOActorImpl::dist(torch::Tensor s)
{                
    torch::Tensor softmax = net->forward(s);    
    return softmax;
}

// Pass back only Q(s, a) for now
torch::Tensor NetPPOActorImpl::forward(torch::Tensor s, torch::Tensor a)
{                
    torch::Tensor softmax = net->forward(s);
    int64_t action = a.item().toLong(); // will this lead to problem?
    assert(action >= 0 && action < action_dim);
    return softmax[action].log();    
}