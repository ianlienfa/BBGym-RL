#include "search_modules/Net/DDPR/NetDDPRQNet.h"

NetDDPRQNetImpl::NetDDPRQNetImpl(int64_t state_dim, int64_t action_dim, Pdd action_range, int64_t num_max_contour, int64_t rnn_hidden_size, int64_t rnn_num_layers)
{
    this->action_dim = action_dim;
    this->state_dim = state_dim;
    this->action_range = action_range;
    net = register_module("Sequential", nn::Sequential(
        nn::Linear(state_dim+rnn_hidden_size+action_dim, 32),
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
    this->rnn_hidden_size = rnn_hidden_size;
    this->rnn_num_layers = rnn_num_layers; 
    // input_size would be 1, and the sequence length would be num_max_contour   
    rnn = register_module("RNN", 
        torch::nn::RNN(nn::RNNOptions(1, rnn_hidden_size).num_layers(rnn_num_layers).batch_first(true))
    );
}


torch::Tensor NetDDPRQNetImpl::forward(torch::Tensor state, torch::Tensor contour_snapshot, torch::Tensor action)
{
    // hidden state init
    torch::Tensor rnn_out;
    int64_t batch_size = state.size(0);
    this->hidden_state = torch::zeros({rnn_num_layers, batch_size, rnn_hidden_size});

    // do rnn encoding on contour snapshot
    std::cout << "contour_snapshot" << std::endl << contour_snapshot << std::endl;
    std::tie(rnn_out, this->hidden_state) = rnn->forward(contour_snapshot, this->hidden_state);

    std::cout << "s.size" << std::endl << state.sizes() << std::endl;
    std::cout << "rnn_out.size" << std::endl << rnn_out.sizes() << std::endl;
    rnn_out = rnn_out.slice(1, rnn_out.size(1)-1, rnn_out.size(1)).reshape({batch_size, rnn_hidden_size});
    state = torch::cat({state, rnn_out}, 1);
    std::cout << "s.size after concat" << std::endl << state.sizes() << std::endl;
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