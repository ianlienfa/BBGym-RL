#include "search_modules/Net/DDPR/NetDDPRActor.h"

NetDDPRActorImpl::NetDDPRActorImpl(const NetDDPROptions& ops)
{    
    net = register_module("Sequential", 
        nn::Sequential(
            nn::Linear(ops.state_dim + ops.max_num_contour, 64),
            nn::ReLU(),
            nn::Linear(64, 32),
            nn::ReLU(),
            nn::Linear(32, (ops.action_range.second - 1) + 2)
        )
    );       

    this->state_dim = ops.state_dim;
    this->action_range = ops.action_range;
    this->split_map[0] = 1;
    this->split_map[1] = 1;
    this->split_map[2] = ops.action_range.second - 1;
    for(int i = 0; i < ops.action_range.second; i++)
    {
        this->arg_softmax_map_arr.push_back(float(i));
    }
    this->arg_softmax_map = torch::from_blob(arg_softmax_map_arr.data(), {1, (long long)(ops.action_range.second-1)}, torch::TensorOptions().dtype(torch::kFloat32));
}

torch::Tensor NetDDPRActorImpl::forward(torch::Tensor s, torch::Tensor contour_snapshot)
{                
    #if TORCH_DEBUG >= -1
    cout << "NetDDPRActorImpl::input" << endl;
    cout << "s: " << s << endl;
    cout << "contour_snapshot: " << contour_snapshot << endl;
    #endif
    const float &limit = (float) action_range.second;
    // initialize hidden state
    const int64_t batch_size = s.sizes()[0];
    const int64_t state_feature_size = s.sizes()[1];
    assertm("state dimension error!", s.sizes().size() == 2);
    assertm("state batch dimension error!", s.sizes()[0] > 0);
    assertm("state feature dimension error!", s.sizes()[1] > 0);

    // do rnn encoding on contour snapshot
    // assert the type of tensor, should be float32
    assertm("contour snapshot tensor type error!", contour_snapshot.dtype() == torch::kFloat32);
    // cout << "contour snapshot input" << endl << contour_snapshot << endl;

    s = torch::cat({s, contour_snapshot}, 1); 
    cout << "s after cat: " << s << endl;   
    torch::Tensor linear_output = net->forward(s);

    #if TORCH_DEBUG >= -1
    cout << "linear_output: " << endl << linear_output << endl;
    #endif    
    auto tensor_bf_split = torch::split_with_sizes(linear_output, this->split_map, -1);

    // prob
    torch::Tensor prob = (tensor_bf_split[0]).sigmoid();
    
    // label_in_num : activation -- 2 / (1 + e^(-x)) - 1
    torch::Tensor label_in_num = (tensor_bf_split[1]);
    label_in_num = 1 + torch::exp(-label_in_num);
    label_in_num = 2 * torch::reciprocal(label_in_num) - 1;

    // // label_softmax
    torch::Tensor label_softmax = (tensor_bf_split[2]);
    label_softmax = label_softmax.softmax(-1);    
    #if TORCH_DEBUG >= -1
    cout << "label_softmax" << endl << label_softmax << endl;
    cout << "label_in_num" << endl << label_in_num << endl;
    cout << "prob: " << endl << prob << endl;
    #endif    
    torch::Tensor output = torch::hstack({prob, label_in_num, label_softmax});
    cout << "output" << endl << output << endl;
    #if TORCH_DEBUG >= 0
    cout << "output" << endl << output << endl;
    #endif
    
    return output;
}