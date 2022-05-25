#include "search_modules/Net/PPO/NetPPOQNet.h"

NetPPOQNetImpl::NetPPOQNetImpl(const NetPPOOptions &ops)
{
    this->action_dim = ops.action_dim;
    this->state_dim = ops.state_dim;
    this->hidden_dim = ops.hidden_dim;
    net = register_module("Sequential", nn::Sequential(
        nn::Linear(ops.state_dim, ops.hidden_dim),
        nn::ReLU(),
        nn::Linear(ops.hidden_dim, ops.hidden_dim),
        nn::ReLU(),
        nn::Linear(ops.hidden_dim, ops.hidden_dim),
        nn::ReLU(),
        nn::Linear(ops.hidden_dim, 1)
    ));

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


torch::Tensor NetPPOQNetImpl::forward(torch::Tensor state)
{
    torch::Tensor output = net->forward(state);
    return output;
}