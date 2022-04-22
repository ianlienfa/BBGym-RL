#include <iostream>
#include <stdexcept>
#include <memory>
#include <fstream>


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

    // ensure the last done operation is updated
    int buffer_size = labeler->buffer->get_size(); 
    int batch_size = (buffer_size < labeler->batch_size) ? buffer_size : labeler->batch_size;
    cout << "buffer_size: " << buffer_size << " batch_size: " << batch_size << endl;
    if(buffer_size) 
    { 
        vector<int> v(batch_size);  
        for(int i = buffer_size - batch_size; i < buffer_size; i++)
            v[i - (buffer_size - batch_size)] = i;        
        RawBatch batch = labeler->buffer->sample(v);         
        labeler->update(batch); 
    } 
    
    cout << "--------Doing Tail updates-------" << endl;
    int tail_updates = labeler->tail_updates;
    while(tail_updates--)
    {
        vector<int> v(batch_size);  
        auto rand_in_range = [=](){
            return (int(std::rand())) % (int(buffer_size));
        };
        generate(v.begin(), v.end(), rand_in_range); 
        RawBatch batch = labeler->buffer->sample(v);         
        labeler->update(batch); 
    }
    cout << "--------Tail updates done-------" << endl;
    // do some more randome updates
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

    // Check if the model file exists
    string qNetPath = QNetPath;
    string piNetPath = PiNetPath;
    string qOptimPath = QOptimPath;
    string piOptimPath = PiOptimPath;
    // string qSchedulerPath = QSchedulerPath;
    // string piSchedulerPath = PiSchedulerPath;
    if(!std::filesystem::exists(qNetPath))
        qNetPath = "";
    if(!std::filesystem::exists(piNetPath))
        piNetPath = "";
    if(!std::filesystem::exists(qOptimPath))
        qOptimPath = "";
    if(!std::filesystem::exists(piOptimPath))
        piOptimPath = "";
    // if(!std::filesystem::exists(qSchedulerPath))
    //     qSchedulerPath = "";
    // if(!std::filesystem::exists(piSchedulerPath))
    //     piSchedulerPath = "";
    std::shared_ptr<DDPRLabeler> labeler = 
        std::make_shared<DDPRLabeler>(
            int64_t(StateInput(OneRjSumCjNode(), OneRjSumCjNode(), OneRjSumCjGraph()).get_state_encoding().size()), 
            3, // prob, noise, floor
            Pdd(-5, 5) /* The output is default at (0, 1), the label will be extend to (-5, 5), 
                            note that the -5 and 5 should not be a feasible output, 
                            this is to preserve the extendibility of labeling */
            ,qNetPath
            ,piNetPath
            ,qOptimPath
            ,piOptimPath
            // ,qSchedulerPath
            // ,piSchedulerPath
        );
    
    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {                        
        string filename(argv[2]);  
        for(int epoch = 1; epoch <= labeler->num_epoch; epoch++)              
        {                
            cerr << "epoch: " << epoch << endl;
            labeler->epoch++;                    
            if(parse_and_init_oneRjSumCj(filename))
            {
                OneRjSumCjSearch searcher(labeler);
                OneRjSumCjBranch brancher;
                OneRjSumCjPrune pruner;
                LowerBound lowerbound;
                OneRjSumCjGraph graph;
                OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                graph = solver.solve(OneRjSumCjNode());  

                #if INF_MODE != 1
                torch::save(labeler->net->q, "../saved_model/qNet.pt");
                torch::save(labeler->net->pi, "../saved_model/piNet.pt"); 
                torch::save((*labeler->optimizer_q), "../saved_model/optimizer_q.pt");
                torch::save((*labeler->optimizer_pi), "../saved_model/optimizer_pi.pt"); 
                // torch::save((*labeler->scheduler_q), "../saved_model/scheduler_q.pt");
                // torch::save((*labeler->scheduler_pi), "../saved_model/scheduler_pi.pt");
                #endif

                // Create file if not exist 
                if(!std::filesystem::exists("../saved_model/q_loss.txt"))
                {
                    std::ofstream outfile("../saved_model/q_loss.txt");
                    outfile.close();
                }
                if(!std::filesystem::exists("../saved_model/pi_loss.txt"))
                {
                    std::ofstream outfile("../saved_model/pi_loss.txt");
                    outfile.close();
                }

                std::ofstream outfile;
                outfile.open("../saved_model/q_loss.txt", std::ios_base::app);    
                for(auto it: labeler->q_mean_loss)
                    outfile << it << ", ";
                outfile.close();

                outfile.open("../saved_model/pi_loss.txt", std::ios_base::app);    
                for(auto it: labeler->pi_mean_loss)
                    outfile << it << ", ";    
                outfile.close();

                // clean up loss_vec
                labeler->q_mean_loss.clear();
                labeler->pi_mean_loss.clear();                

                // stage save
                if(labeler->num_epoch > 5 && epoch % (int(labeler->num_epoch / 5)) == 0)
                {
                    string cmd = "cp ../saved_model/qNet.pt ../saved_model/qNet" + std::to_string(epoch) + ".pt";
                    exec(cmd.c_str());
                    cmd = "cp ../saved_model/piNet.pt ../saved_model/piNet" + std::to_string(epoch) + ".pt";
                    exec(cmd.c_str());
                }
            }

            #if INF_MODE == 1
            epoch = labeler->num_epoch;
            #endif
        }
    }
    if(argc == 2)
    {        
        // read problem
        #define INSTANCE_NUM 42
        srand(time(NULL));
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

                #if INF_MODE != 1
                torch::save(labeler->net->q, "../saved_model/qNet.pt");
                torch::save(labeler->net->pi, "../saved_model/piNet.pt"); 
                torch::save((*labeler->optimizer_q), "../saved_model/optimizer_q.pt");
                torch::save((*labeler->optimizer_pi), "../saved_model/optimizer_pi.pt"); 
                // torch::save((*labeler->scheduler_q), "../saved_model/scheduler_q.pt");
                // torch::save((*labeler->scheduler_pi), "../saved_model/scheduler_pi.pt");
                #endif

                // Create file if not exist 
                if(!std::filesystem::exists("../saved_model/q_loss.txt"))
                {
                    std::ofstream outfile("../saved_model/q_loss.txt");
                    outfile.close();
                }
                if(!std::filesystem::exists("../saved_model/pi_loss.txt"))
                {
                    std::ofstream outfile("../saved_model/pi_loss.txt");
                    outfile.close();
                }

                std::ofstream outfile;
                outfile.open("../saved_model/q_loss.txt", std::ios_base::app);    
                for(auto it: labeler->q_loss_vec)
                    outfile << it << ", ";
                outfile.close();

                outfile.open("../saved_model/pi_loss.txt", std::ios_base::app);    
                for(auto it: labeler->pi_loss_vec)
                    outfile << it << ", ";    
                outfile.close();

                // clean up loss_vec
                labeler->q_loss_vec.clear();
                labeler->pi_loss_vec.clear();
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
