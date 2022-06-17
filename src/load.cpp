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
    using std::list;
    list<int> li = {1, 2, 3};
    list<int>::iterator it = li.begin();    
    li.emplace(it, 4);
    for(auto it: li)
        cout << it << " ";
}