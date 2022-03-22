#include <iostream>

#include "user_def/oneRjSumCj_engine.h"
#include "user_def/oneRjSumCjPrune.h"
#include "problem_parser/problemParser.h"
#include "search_modules/Net/DDPR/NetDDPR.h"

int main(int argc, char* argv[])
{        

    OneRjSumCjPrune::prune_funcs = {
        prune__OneRjSumCj__LU_AND_SAL__Theorem1
    };
    OneRjSumCjPrune::safe_prune_funcs = {
        pruneIncumbentCmpr
    };
        
    if (argc != 2)
    {
        if(parse_and_init_oneRjSumCj())
        {
            OneRjSumCj_engine solver; 
            OneRjSumCjGraph graph;
            graph = solver.solve(OneRjSumCjNode());  
        }
    }
    else
    {
            // read problem
        #define INSTANCE_NUM 5
        srand(0);
        InputHandler inputHandler((string(argv[1])));
        string filepath;
        int instance_idx = INSTANCE_NUM;
        while(instance_idx--)
        {
            int step_size = rand() % 3;            
            do
            {
                filepath = inputHandler.getNextFileName();  
                if(filepath.empty())
                    inputHandler.reset();
            }while(step_size--);
            if(parse_and_init_oneRjSumCj(filepath))
            {
                std::shared_ptr<DDPRLabeler> labeler = std::make_shared<DDPRLabeler>(int64_t(StateInput(OneRjSumCjNode(), OneRjSumCjNode()).flatten_and_norm().size()), 1, Pdd(0.0, 10.0));
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

    /* For validation */
    // int min_obj = INT_MAX;
    // vector<int> min_seq;
    // vector<int> v;
    // for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++)
    //     v.push_back(i);
    // do{
    //     // for(auto it: v)
    //     //     cout << it << " ";
    //     // cout << endl;
    //     pair<int, int> p = OneRjSumCjNode::getObj(v);
    //     if(min_obj > p.first){
    //         min_obj = p.first;
    //         min_seq = v;
    //     }
    // }while(next_permutation(v.begin(), v.end()));
    
    // cout << "min_obj: " << endl;
    // for(auto it: min_seq)
    //     cout << it << " ";
    // cout << endl;
    // cout << min_obj << endl;
}
