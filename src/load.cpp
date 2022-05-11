#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include "util/types.h"
using std::vector, std::string, std::cout, std::endl;

int main()
{
    for(int i = 0; i < 10; i++)
    {
        cout << std::setw(5) << (BB_RAND() % 100) << endl; 
    }
}

