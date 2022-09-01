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
            nn::Linear(hidden_dim, hidden_dim),
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
            torch::manual_seed(RANDOM_SEED);
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
    torch::NoGradGuard no_grad;
    auto softmax = net->forward(s);    
    return softmax;
}

// Pass back only Q(s, a) for now
torch::Tensor NetPPOActorImpl::forward(torch::Tensor s, torch::Tensor a)
{                    
    torch::Tensor softmax = net->forward(s);
    torch::Tensor idx = a.argmax(1);
    idx = idx.unsqueeze(1);
    torch::Tensor softmax_val = torch::gather(softmax, 1, idx);

    // for debug
    #if VALIDATION_LEVEL == validation_level_HIGH
    std::vector<int64_t> action(idx.data_ptr<int64_t>(), idx.data_ptr<int64_t>() + idx.numel());
    for(int i = 0; i < action.size(); i++)
    {
        assertm("action not ok", action[i] >= 0 && action[i] < action_dim);
    } //
    #endif

    return softmax_val.log();    
}