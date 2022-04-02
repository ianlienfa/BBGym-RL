#include <iostream>
#include <vector>
#include <torch/torch.h>

using std::vector, std::string, std::cout, std::endl;

struct NetTestImpl: torch::nn::Module
{

    torch::nn::Sequential net{nullptr};

    NetTestImpl(){
        net = register_module("Seq", torch::nn::Sequential(
            torch::nn::Linear(1, 2),
            torch::nn::ReLU(),
            torch::nn::Linear(2, 1)
        ));
    };

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }
};
TORCH_MODULE(NetTest);

int main()
{
    vector<int> shape = {1, 3, 224, 224};
    NetTest test;
    torch::Tensor tensor1 = torch::from_blob(shape.data(), {1, 4}, torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor tensor2 = tensor1.clone();

    torch::Tensor out = test(torch::rand({1, 1}));
    cout << out << endl;
    // cout << "Printing layer weights of " << test->name() << endl;
    // for (auto& p : test->net->named_parameters(true)) {

    //     torch::NoGradGuard no_grad;

    //     // Access name.
    //     std::cout << p.key() << std::endl;

    //     // Access weigth and bias.
    //     std::cout << p.value() << std::endl;
    // }

    const auto param = test->parameters(true);
    auto it = param.begin();
    auto it_end = param.end();
    while(it != it_end){    
        (*it).print();
        it++;
    }
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
