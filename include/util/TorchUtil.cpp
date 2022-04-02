#include "TorchUtil.h"

void layer_weight_print(const torch::nn::Module &net)
{    
    using namespace std;
    cout << "Printing layer weights of " << net.name() << endl;
    for (auto& p : net.named_parameters()) {

        torch::NoGradGuard no_grad;

        // Access name.
        std::cout << p.key() << std::endl;

        // Access weigth and bias.
        std::cout << p.value() << std::endl;
    }
};

