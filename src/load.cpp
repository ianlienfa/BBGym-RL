#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include "util/PriorityQueue.h"
using std::vector, std::string, std::cout, std::endl;

int main()
{
    torch::Tensor t = torch::tensor({{1}, {2}, {3}}, torch::kFloat32);
    auto mean = t.mean(0);
    std::cout << mean << std::endl;
}