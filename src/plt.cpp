#include <iostream>
#include <iomanip>
#include "third_party/matplotlibcpp/include/matplotlibcpp.h"
namespace plt = matplotlibcpp;
using namespace std;

int main() {
    plt::plot({1,3,2,4});
    plt::savefig("../case/test.png");
}
