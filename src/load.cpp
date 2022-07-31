#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include <list>
#include "util/PriorityQueue.h"
#include "problem_parser/problemParser.h"
using std::vector, std::string, std::cout, std::endl, std::cerr;


int main(int argc, char* argv[])
{

    if (argc >= 3 && !(strcmp(argv[1], "-d")))
    {        
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;
        
        // read problem
        InputHandler inputHandler((string(argv[2])));
        InputHandler inputHandler_test((string(argv[2])) + "/test");
        string filepath;
        int step_size = 1;            
        for(int i = 0; i < 100; i++)
        {
            int rand_jump = rand() % 5 + 1;
            for(int j = 0; j < rand_jump; j++)
            {
                filepath = inputHandler.getNextFileName();              
                cout << "processing: " << filepath << endl;
            }
        }
    }
}
