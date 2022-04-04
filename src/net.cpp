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

/* 
    SOLVE_CALLBACK() is called at each search-branch-prune iteration,
    we call network update at some steps.
*/    

void solveCallbackImpl(void* engine_ptr)
{     
    #if INF_MODE != 1
    OneRjSumCj_engine &engine = *(static_cast<OneRjSumCj_engine*>(engine_ptr));
    auto &labeler = engine.searcher.labeler;
    if(labeler->step % labeler->update_freq == 0 && labeler->buffer->get_size() > labeler->batch_size) 
    { 
        int buffer_size = labeler->buffer->get_size(); 
        vector<int> v(labeler->batch_size);  
        auto rand_in_range = [=](){
            return (int(std::rand())) % (int(buffer_size));
        };
        generate(v.begin(), v.end(), rand_in_range); 
        RawBatch batch = labeler->buffer->sample(v);         
        labeler->update(batch); 
    } 
    #endif
}

void optimalFoundCallbackImpl(void* engine_ptr)
{
    #if INF_MODE != 1
    cout << "One last update" << endl;
    OneRjSumCj_engine &engine = *(static_cast<OneRjSumCj_engine*>(engine_ptr));
    auto &labeler = engine.searcher.labeler;
    if(engine.graph.optimal_found != true)
        throw std::runtime_error("Optimal not found but optimal Found callback called!");
    int buffer_size = labeler->buffer->get_size(); 
    if(buffer_size) 
    { 
        int batch_size = (buffer_size < labeler->batch_size) ? buffer_size : labeler->batch_size;
        vector<int> v(batch_size);  
        for(int i = buffer_size - batch_size; i < buffer_size; i++)
            v[i - (buffer_size - batch_size)] = i;        
        RawBatch batch = labeler->buffer->sample(v);         
        // For testing
        bool has_done = false;
        for(auto it : get<4>(batch))
            if(it == 0)
                has_done = true;
        if(!has_done)
            throw std::runtime_error("Optimal found but optimal batch has no done!");
        labeler->update(batch); 
    } 
    #endif
}

int main(int argc, char* argv[])
{        

    OneRjSumCjPrune::prune_funcs = {
        prune__OneRjSumCj__LU_AND_SAL__Theorem1
    };
    OneRjSumCjPrune::safe_prune_funcs = {
        pruneIncumbentCmpr
    };

    string qNetPath = QNetPath;
    string piNetPath = PiNetPath;

    std::shared_ptr<DDPRLabeler> labeler = 
        std::make_shared<DDPRLabeler>(
            int64_t(StateInput(OneRjSumCjNode(), OneRjSumCjNode(), OneRjSumCjGraph()).get_state_encoding().size()), 
            1, 
            Pdd(-5, 5) /* The output is default at (0, 1), the label will be extend to (-5, 5), 
                            note that the -5 and 5 should not be a feasible output, 
                            this is to preserve the extendibility of labeling */
            // ,qNetPath
            // ,piNetPath
        );
    
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
        #define INSTANCE_NUM 10
        srand(0);
        InputHandler inputHandler((string(argv[1])));
        string filepath;
        int instance_idx = INSTANCE_NUM;
        cout << "instance number: " << instance_idx << endl;
    
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
                string cmd = "echo \"\" >> fileSearched.txt";
                exec(cmd.c_str());
                cmd = "echo '" + filepath + "' >> " + "fileSearched.txt";
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

    // cout << "q_loss: " << endl << "[ ";
    // for(auto it: labeler->q_loss_vec)
    //     cout << it << ", " << endl;
    // cout << " ]" << endl;

    // cout << "pi_loss: " << endl << "[ ";
    // for(auto it: labeler->pi_loss_vec)
    //     cout << it << ", " << endl;
    // cout << " ]" << endl;

    #if INF_MODE != 1
    torch::save(labeler->net->q, "../saved_model/qNet.pt");
    torch::save(labeler->net->pi, "../saved_model/piNet.pt"); 
    #endif

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
