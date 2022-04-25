#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(int64_t state_dim, Pdd action_range)
{
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(state_dim, 64),
            nn::ReLU(),
            // nn::Linear(64, 256),
            // nn::ReLU(),
            // nn::Linear(256, 64),
            // nn::ReLU(),
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
    for(int i = 0; i < action_range.second; i++)
    {
        this->arg_softmax_map_arr.push_back(float(i));
    }
    this->arg_softmax_map = torch::from_blob(arg_softmax_map_arr.data(), {1, (long long)(action_range.second-1)}, torch::TensorOptions().dtype(torch::kFloat32));
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
    #if TORCH_DEBUG >= 0
    cout << "prob: " << endl << prob << endl;
    #endif
    
    // label_in_num
    torch::Tensor label_in_num = (tensor_bf_split[1]).tanh(); 

    // // label_softmax
    torch::Tensor label_softmax = (tensor_bf_split[2]);
    label_softmax = label_softmax.softmax(-1);    
    #if TORCH_DEBUG >= 0
    cout << "label_softmax" << endl << label_softmax << endl;
    cout << "arg_softmax_map" << endl << this->arg_softmax_map << endl;
    #endif    

    // label_softmax = label_softmax.mul(arg_softmax_map);
    // #if TORCH_DEBUG >= 0    
    // if(label_softmax.grad_fn() != NULL)
    //     cout << "label_softmax grad_fn: " << label_softmax.grad_fn()->name() << endl;
    // cout << "label_softmax after mul" << endl << label_softmax << endl;
    // #endif

    // label_softmax = label_softmax.sum(-1).unsqueeze(-1).floor().add(1.0);
    // #if TORCH_DEBUG >= 0
    // cout << "label_softmax after sum" << endl << label_softmax << endl;
    // #endif

    torch::Tensor output = torch::hstack({prob, label_in_num, label_softmax});

    #if TORCH_DEBUG >= 0
    cout << "output" << endl << output << endl;
    #endif

    return output;
}