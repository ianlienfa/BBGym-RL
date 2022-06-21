#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include <list>
#include "util/PriorityQueue.h"
using std::vector, std::string, std::cout, std::endl;

int main()
{
    torch::Tensor dist = torch::tensor({0.1, 0.2, 0.6, 0.1});
    vector<int> indices(4, 0);
    for(int i = 0; i < 1000; i++)
    {
        int64_t a = torch::multinomial(dist, 1).item<int64_t>();        
        indices[a]++;
    }
    for(auto it: indices)
        it = it/1000;
    cout << indices << endl;
    
}