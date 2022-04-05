#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "search_modules/Net/DDPR/NetDDPRActor.h"

using std::vector, std::string, std::cout, std::endl;

struct NetTestImpl: torch::nn::Module
{

    torch::nn::Sequential net{nullptr};
    torch::nn::Sigmoid sig1{nullptr};
    torch::nn::Sigmoid sig2{nullptr};

    NetTestImpl(){
        net = register_module("Seq", torch::nn::Sequential(
            torch::nn::Linear(4, 16),
            torch::nn::ReLU(),
            torch::nn::Linear(16, 16),
            torch::nn::ReLU(),
            torch::nn::Linear(16, 1)
        ));
    };

    torch::Tensor forward(torch::Tensor x) {                
        torch::Tensor output = net->forward(x);
        return output;
    }
};
TORCH_MODULE(NetTest);

int main()
{
    vector<float> s1 = {1, 0.3, 4, 3, 5, 6, 1, 2};
    vector<float> s2 = 
    {
        0.2222,  0.2397,  0.0104,  0.2879,  0.2222,  0.2397,  0.0104,  0.2879,
        0.1111,  0.2673,  0.0088,  0.3638,  0.1111,  0.2673,  0.0088,  0.3638
    };

    torch::Tensor in1 = torch::from_blob(s1.data(), {1, 8});
    torch::Tensor in2 = torch::from_blob(s2.data(), {2, 8});

    NetDDPRActor net{nullptr};
    net = NetDDPRActor(8, std::make_pair(-5, 5));
    torch::Tensor out1 = net->forward(in1);
    torch::Tensor out2 = net->forward(in2);
    cout << "out1: " << endl << out1 << endl;
    cout << "out2: " << endl << out2 << endl;
    
    float out1_data = out1.item<float>();
    cout << out1_data << endl;
}

// #include "util/TorchUtil.h"
// #include "search_modules/Net/DDPR/NetDDPR.h"

// float min_y = -1.93109e+06, max_y = 1.77644e+06;
// float min_x = 2, max_x = 98;
// float min_z = -99, max_z = 96;

// float denormalize(float network_output)
// {
//     if(max_y != NULL && min_y != NULL)
//     {
//         return network_output * (max_y - min_y) + min_y;
//     }        
//     else
//     {
//       return 0;
//     }
// }

// float normalize(float in, float min_val, float max_val)
// {
//     float denom = (max_val - min_val) ? (max_val - min_val) : 1;
//     return (in - min_val) / denom;
// }

// int main(int argc, char *argv[])
// {
//     NetDDPRQNet net(1, 1);
//     torch::load(net, "net.pt");  
//     if(argc != 3)
//         return 0;
//     {
//         torch::NoGradGuard no_grad;
//         float x[] = {normalize(stof(string(argv[1])), min_x, max_x)};
//         float z[] = {normalize(stof(string(argv[2])), min_z, max_z)};
//         auto state = torch::from_blob(x, {1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//         auto action = torch::from_blob(z, {1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//         cout << "state: " << state << endl;
//         cout << "action: " << action << endl;
//         auto test_prediction = net->forward(state, action);
//         cout << test_prediction << endl;
//         // auto out_iter = test_prediction.accessor<float, 1>();
//         // cout << "prediction: " << out_iter[0] << endl;
//         // cout << "prediction denorm: " << denormalize(out_iter[0]) << endl;
//         cout << "prediction: " << denormalize(test_prediction.item<float>()) << endl;
//     } 
// }
