#include <iostream>
#include <vector>
#include <torch/torch.h>

using std::vector, std::string, std::cout, std::endl;

struct NetTestImpl: torch::nn::Module
{

    torch::nn::Sequential net{nullptr};
    torch::nn::Sigmoid sig1{nullptr};
    torch::nn::Sigmoid sig2{nullptr};

    NetTestImpl(){
        net = register_module("Seq", torch::nn::Sequential(
            torch::nn::Linear(1, 2),
            torch::nn::ReLU(),
            torch::nn::Linear(2, 2),
            torch::nn::Sigmoid()
        ));
    };

    torch::Tensor forward(torch::Tensor x) {                
        const int64_t &limit = 5;
        torch::Tensor output = net->forward(x);
        cout << "output: " << output << endl;
        auto action_and_prob = output.unbind(1);
        output = action_and_prob[0] * limit;
        torch::Tensor prob = action_and_prob[1];
        output = output.where(prob > 0.5, output.floor());    
        cout << "prob: " << prob << endl;
        cout << "output: " << output << endl;
        return output;
    }
};
TORCH_MODULE(NetTest);

int main()
{
    // vector<int> shape = {1, 3, 224, 224};
    // NetTest test;
    // torch::Tensor tensor1 = torch::from_blob(shape.data(), {1, 4}, torch::TensorOptions().dtype(torch::kInt32));
    // torch::Tensor tensor2 = tensor1.clone();

    // torch::Tensor out = test(torch::rand({1, 1}));
    // cout << out << endl;

    vector<float> vec = {0.51, 0.45};
    torch::Tensor x = torch::from_blob(vec.data(), {1, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    cout << "x: " << x << endl;
    torch::Tensor output = x;
    x.requires_grad_(true);
    cout << "output: " << output << endl;
    auto action_and_prob = output.unbind(1);
    output = action_and_prob[0] * 5;
    torch::Tensor prob = action_and_prob[1];
    cout << "prob: " << prob << endl;
    cout << "output: " << output << endl;
    output.backward();
    prob.backward();
    cout << x.grad() << endl;

    torch::Tensor output_where = output.where(prob < 0.5, output.floor()); 
    cout << "output after where: " << output_where << endl;   
    cout << output_where << endl;
    output_where.backward();
    cout << x.grad() << endl;
    

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
