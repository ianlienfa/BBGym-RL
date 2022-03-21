#include <iostream>
#include <torch/torch.h>
#include "util/TorchUtil.h"
#include "search_modules/Net/DDPR/NetDDPR.h"

float min_y = -1.93109e+06, max_y = 1.77644e+06;
float min_x = 2, max_x = 98;
float min_z = -99, max_z = 96;

float denormalize(float network_output)
{
    if(max_y != NULL && min_y != NULL)
    {
        return network_output * (max_y - min_y) + min_y;
    }        
    else
    {
      return 0;
    }
}

float normalize(float in, float min_val, float max_val)
{
    float denom = (max_val - min_val) ? (max_val - min_val) : 1;
    return (in - min_val) / denom;
}

int main(int argc, char *argv[])
{
    NetDDPRQNet net(1, 1);
    torch::load(net, "net.pt");  
    if(argc != 3)
        return 0;
    {
        torch::NoGradGuard no_grad;
        float x[] = {normalize(stof(string(argv[1])), min_x, max_x)};
        float z[] = {normalize(stof(string(argv[2])), min_z, max_z)};
        auto state = torch::from_blob(x, {1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        auto action = torch::from_blob(z, {1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        cout << "state: " << state << endl;
        cout << "action: " << action << endl;
        auto test_prediction = net->forward(state, action);
        cout << test_prediction << endl;
        // auto out_iter = test_prediction.accessor<float, 1>();
        // cout << "prediction: " << out_iter[0] << endl;
        // cout << "prediction denorm: " << denormalize(out_iter[0]) << endl;
        cout << "prediction: " << denormalize(test_prediction.item<float>()) << endl;
    } 
}