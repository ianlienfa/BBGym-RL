#include <iostream>
#include <stdexcept>
#include <memory>


#include "user_def/oneRjSumCj_engine.h"
#include "user_def/oneRjSumCjPrune.h"
#include "problem_parser/problemParser.h"
#include "search_modules/Net/DDPR/NetDDPR.h"


std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main(int argc, char* argv[])
{        

    OneRjSumCjPrune::prune_funcs = {
        prune__OneRjSumCj__LU_AND_SAL__Theorem1
    };
    OneRjSumCjPrune::safe_prune_funcs = {
        pruneIncumbentCmpr
    };

    OneRjSumCjGraph dummy_graph;
    OneRjSumCjNode dummy_node;
    auto rand_in_range = [](int max_num){
        return std::rand() % max_num;
    };

    PlainLabeler plainLabeler;
    std::shared_ptr<DDPRLabeler> labeler = 
        std::make_shared<DDPRLabeler>(
            int64_t(StateInput(dummy_node, dummy_node, dummy_graph).get_state_encoding().size()), 
            1, 
            Pdd(-5, 5) /* The output is default at (0, 1), the label will be extend to (-5, 5), 
                            note that the -5 and 5 should not be a feasible output, 
                            this is to preserve the extendibility of labeling */
        );

    /* 
        SOLVE_CALLBACK() is called at each search-branch-prune iteration,
        we call network update at some steps.
    */
    #undef SOLVE_CALLBACK
    #define SOLVE_CALLBACK() \
    { \
        if(labeler->step % labeler->update_freq == 0) \
        { \
            int buffer_size = labeler->buffer->get_size(); \
            vector<int> v(labeler->buffer_size);  \
            generate(v.begin(), v.end(), rand_in_range(buffer_size)); \
            Batch batch = labeler->buffer->get(v); \
            labeler->update(batch); \
        } \
    }      
        
    if (argc != 2)
    {
        if(parse_and_init_oneRjSumCj())
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
                string cmd = "echo '" + filepath + "' >> " + "fileSearched.txt";
                exec(cmd.c_str());
                OneRjSumCjSearch searcher(labeler);
                OneRjSumCjBranch brancher;
                OneRjSumCjPrune pruner;
                LowerBound lowerbound;
                OneRjSumCjGraph graph;
                OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                graph = solver.solve(OneRjSumCjNode());  
            }                                          
            labeler->epoch++;                    
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
