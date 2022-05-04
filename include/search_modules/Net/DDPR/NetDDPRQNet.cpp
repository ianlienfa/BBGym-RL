#include "search_modules/Net/DDPR/NetDDPRQNet.h"

NetDDPRQNetImpl::NetDDPRQNetImpl(const NetDDPROptions &ops)
{
    this->action_dim = ops.action_dim;
    this->state_dim = ops.state_dim;
    this->action_range = ops.action_range;    
    net = register_module("Sequential", nn::Sequential(
        nn::Linear(ops.state_dim + ops.action_dim + ops.max_num_contour, 32),
        nn::ReLU(),
        nn::Linear(32, 64),
        nn::ReLU(),
        nn::Linear(64, 32),
        nn::ReLU(),
        nn::Linear(32, 1)
    ));
    // {
    //     torch::NoGradGuard no_grad;
    //     // weight init
    //     auto initialize_weights_norm = [](nn::Module& module) {
    //         torch::NoGradGuard no_grad;
    //         if (auto* linear = module.as<nn::Linear>()) {
    //             torch::nn::init::xavier_normal_(linear->weight);
    //             torch::nn::init::constant_(linear->bias, 0.01);
    //         }
    //     };
    //     this->apply(initialize_weights_norm);
    // }
}


torch::Tensor NetDDPRQNetImpl::forward(torch::Tensor state, torch::Tensor contour_snapshot, torch::Tensor action)
{
    int64_t batch_size = state.size(0);

    state = torch::cat({state, contour_snapshot}, 1);
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