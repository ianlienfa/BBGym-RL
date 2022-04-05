#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(int64_t state_dim, Pdd action_range)
{
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, 64),
            nn::ReLU(),
            nn::Linear(64, 256),
            nn::ReLU(),
            nn::Linear(256, 64),
            nn::ReLU(),
            nn::Linear(64, 32),
            nn::ReLU(),
            nn::Linear(32, (action_range.second - 1) + 2)
        )
    );       
    this->state_dim = state_dim;
    this->action_range = action_range;
    this->split_map[0] = 1;
    this->split_map[1] = 1;
    this->split_map[2] = action_range.second - 1;
}

torch::Tensor NetDDPRActorImpl::forward(torch::Tensor s)
{            
    #if TORCH_DEBUG >= 0
    cout << "NetDDPRActorImpl::input" << endl;
    cout << s << endl;
    #endif
    const float &limit = (float) action_range.second;
    torch::Tensor linear_output = net->forward(s);

    #if TORCH_DEBUG >= 0
    cout << "linear_output: " << endl << linear_output << endl;
    #endif
    auto tensor_bf_split = torch::split_with_sizes(linear_output, this->split_map, -1);

    // prob
    torch::Tensor prob = (tensor_bf_split[0]).sigmoid();

    // label_in_num
    torch::Tensor raw_action = (tensor_bf_split[1]);
    torch::Tensor extend_sigmoid_action = torch::exp(raw_action * (-limit)) + 1;    
    extend_sigmoid_action = extend_sigmoid_action.reciprocal();
    torch::Tensor label_in_num = extend_sigmoid_action * limit;

    // label_softmax
    torch::Tensor label_softmax = (tensor_bf_split[2]);
    label_softmax = label_softmax.softmax(-1);    
    #if TORCH_DEBUG >= 0
    cout << "label_softmax" << endl << label_softmax << endl;
    #endif
    torch::Tensor output = (get<1>(label_softmax.max(-1))).toType(torch::kFloat32).unsqueeze(1);

    #if TORCH_DEBUG >= 0
    cout << "prob: " << endl << prob << endl;
    cout << "label_in_num: " << endl << label_in_num << endl;
    cout << "softmax output: " << endl << output << endl; 
    #endif
    output = torch::where(prob > 0.5, output, label_in_num);
    #if TORCH_DEBUG >= 0
    cout << "label output: " << endl << output << endl; 
    #endif
    return output;
}