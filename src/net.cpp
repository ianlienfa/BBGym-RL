#include <iostream>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <chrono>

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
    
}

void optimalFoundCallbackImpl(void* engine_ptr)
{
    OneRjSumCj_engine &engine = *(static_cast<OneRjSumCj_engine*>(engine_ptr));
    auto &labeler = engine.searcher.labeler;
    if(labeler->get_labeler_state() == PPO::PPOLabeler::LabelerState::TRAIN_RUNNING)
    {
        PPO::SampleBatch batch = labeler->buffer->get();
        labeler->update(batch);
    }
}

int main(int argc, char* argv[])
{            
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
    if(!std::filesystem::exists(qNetPath))
        qNetPath = "";
    if(!std::filesystem::exists(piNetPath))
        piNetPath = "";
    if(!std::filesystem::exists(qOptimPath))
        qOptimPath = "";
    if(!std::filesystem::exists(piOptimPath))
        piOptimPath = "";

    const int64_t max_num_contour = 10;
    
    std::shared_ptr<PPO::PPOLabeler> labeler = 
        std::make_shared<PPO::PPOLabeler>(
            PPO::PPOLabelerOptions()                
                .state_dim(int64_t(PPO::StateInput(OneRjSumCjNode(), OneRjSumCjNode(), OneRjSumCjGraph().set_max_size(max_num_contour)).get_state_encoding(max_num_contour).size()))
                .action_dim(4)
                .load_q_path(qNetPath)
                .load_pi_path(piNetPath)
                .q_optim_path(qOptimPath)
                .pi_optim_path(piOptimPath)
                .max_num_contour(max_num_contour)            
        );
    
    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {              
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;
                      
        string filename(argv[2]);  
        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {                
            cerr << "epoch: " << epoch << endl;
            labeler->epoch()++;                    
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
                if(labeler->opt.num_epoch() > 5 && epoch % (int(labeler->opt.num_epoch() / 5)) == 0)
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
    // if(argc == 2)
    // {        
    //     // read problem
    //     #define INSTANCE_NUM 42
    //     InputHandler inputHandler((string(argv[1])));
    //     string filepath;
    //     int instance_idx = INSTANCE_NUM;
    //     cout << "instance number: " << instance_idx << endl;
    
    //     while(instance_idx--)
    //     {
    //         int step_size = rand() % 3;            
    //         do
    //         {
    //             filepath = inputHandler.getNextFileName();  
    //             if(filepath.empty())
    //                 inputHandler.reset();
    //         }while(step_size--);
    //         if(parse_and_init_oneRjSumCj(filepath))
    //         {
    //             string cmd = "echo \"\" >> fileSearched.txt";
    //             exec(cmd.c_str());
    //             cmd = "echo '" + filepath + "' >> " + "fileSearched.txt";
    //             exec(cmd.c_str());
    //             OneRjSumCjSearch searcher(labeler);
    //             OneRjSumCjBranch brancher;
    //             OneRjSumCjPrune pruner;
    //             LowerBound lowerbound;
    //             OneRjSumCjGraph graph;
    //             OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
    //             graph = solver.solve(OneRjSumCjNode());  

    //             #if INF_MODE != 1
    //             torch::save(labeler->net->q, "../saved_model/qNet.pt");
    //             torch::save(labeler->net->pi, "../saved_model/piNet.pt"); 
    //             torch::save((*labeler->optimizer_q), "../saved_model/optimizer_q.pt");
    //             torch::save((*labeler->optimizer_pi), "../saved_model/optimizer_pi.pt"); 
    //             #endif

    //             // Create file if not exist 
    //             if(!std::filesystem::exists("../saved_model/q_loss.txt"))
    //             {
    //                 std::ofstream outfile("../saved_model/q_loss.txt");
    //                 outfile.close();
    //             }
    //             if(!std::filesystem::exists("../saved_model/pi_loss.txt"))
    //             {
    //                 std::ofstream outfile("../saved_model/pi_loss.txt");
    //                 outfile.close();
    //             }

    //             std::ofstream outfile;
    //             outfile.open("../saved_model/q_loss.txt", std::ios_base::app);    
    //             for(auto it: labeler->q_loss_vec)
    //                 outfile << it << ", ";
    //             outfile.close();

    //             outfile.open("../saved_model/pi_loss.txt", std::ios_base::app);    
    //             for(auto it: labeler->pi_loss_vec)
    //                 outfile << it << ", ";    
    //             outfile.close();

    //             // clean up loss_vec
    //             labeler->q_loss_vec.clear();
    //             labeler->pi_loss_vec.clear();
    //         }                                          
    //         labeler->epoch++;                    
    //     }         
    // }

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
