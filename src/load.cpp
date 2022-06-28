#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <iomanip>
#include "util/types.h"
#include "user_def/oneRjSumCj_engine.h"
#include "user_def/oneRjSumCjPrune.h"
#include "problem_parser/problemParser.h"
using std::vector, std::string, std::cout, std::endl;
using std::pair;

void solveCallbackImpl(void* engine_ptr){}
void optimalFoundCallbackImpl(void* engine_ptr){}


std::shared_ptr<PlainLabeler> labeler = std::make_shared<PlainLabeler>();


int main(int argc, char* argv[])
{
    OneRjSumCjPrune::prune_funcs = {
    prune__OneRjSumCj__LU_AND_SAL__Theorem1
    };
    OneRjSumCjPrune::safe_prune_funcs = {
        pruneIncumbentCmpr
    };
    string filename = "";
    // parse command line arguments
    if(argc < 2)
    {
        cout << "Usage: ./net_ddpr -f <problem_file>" << endl;
        exit(1);
    }
    else
    {
        vector<pair<string, vector<string>>> commands;
        for(int i = 0; i < argc; i++)
        {
            pair<string, vector<string>> command;
            if(string(argv[i])[0] == '-')
            {
                command.first = string(argv[i]);
            }
            else
            {
                command.second.push_back(string(argv[i]));
            }
        }

        // parse command 
        for(auto &command : commands)
        {
            if(command.first == "-f")
            {
                if(command.second.size() != 1)
                {
                    cout << "Usage: ./net_ddpr -f <problem_file>" << endl;
                    exit(1);
                }
                filename = command.second[0];
            }
            else if(command.first == "-v")
            {
                for(auto &hyper_param : command.second)
                {
                    string name = hyper_param.substr(0, hyper_param.find("="));
                    float value = std::stof(hyper_param.substr(hyper_param.find("=") + 1));
                }
            }
            else
            {
                cout << "not supported command: '" << command.first << "'" << endl;
            }

        }
    }

    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {              
        int rand_seed = RANDOM_SEED;
        cerr << "Random seed: " << rand_seed << endl;
        torch::manual_seed(RANDOM_SEED);    
                      
        string filename(argv[2]);  
        if(parse_and_init_oneRjSumCj(filename))
        {
            OneRjSumCjSearch searcher(labeler);
            OneRjSumCjBranch brancher;
            OneRjSumCjPrune pruner;
            LowerBound lowerbound;
            OneRjSumCjGraph graph;
            OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
            graph = solver.solve(OneRjSumCjNode());
        }  
    }
}

