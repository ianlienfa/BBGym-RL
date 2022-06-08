#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include "util/types.h"
#include "util/PlacementList.h"
using std::vector, std::string, std::cout, std::endl;

int main()
{
    PlacementList<int, int> pl;
    pl.init([](const int& i1, const int& i2) { return i1 < i2; });
    pl.place(1);    
    pl.insert_and_place(2);
    pl.print();
    
    pl.left();
    pl.left();
    pl.right();
    pl.print();

    pl.step_forward();    
    pl.step_forward();    
    int i = pl.pop_from_current();
    pl.print();
    pl.place(3);
    pl.print();

    pl.erase_contour();
    pl.print();
}