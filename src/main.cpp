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

void updateCallbackImpl(void* engine_ptr)
{
    // update
    OneRjSumCj_engine &engine = *(static_cast<OneRjSumCj_engine*>(engine_ptr));
    auto &labeler = engine.searcher.labeler;
    auto current_labeler_state = labeler->get_labeler_state();
    if(current_labeler_state == PPO::PPOLabeler::LabelerState::TRAIN_RUNNING)
    {
        PPO::SampleBatch batch = labeler->buffer->get();
        labeler->update(batch);
    }

    // step reset
    engine.searcher.labeler->step() = 0;

    // reward tracking
    if(current_labeler_state == PPO::PPOLabeler::LabelerState::INFERENCE)
    {
        if(engine.searcher.labeler->avg_inf_reward){ 
            engine.searcher.labeler->avg_inf_reward = engine.searcher.labeler->avg_inf_reward * 0.9 + engine.searcher.labeler->accu_reward * 0.1;
        }
        else{        
            engine.searcher.labeler->avg_inf_reward = engine.searcher.labeler->accu_reward;
        }
        std::ofstream outfile;
        outfile.open("../saved_model/inf_rewards.txt", std::ios_base::app);  
        outfile << engine.searcher.labeler->avg_inf_reward << ", ";  
        outfile.close();
    }
    else
    {
        if(engine.searcher.labeler->avg_reward){ 
            engine.searcher.labeler->avg_reward = engine.searcher.labeler->avg_reward * 0.9 + engine.searcher.labeler->accu_reward * 0.1;
        }
        else{        
            engine.searcher.labeler->avg_reward = engine.searcher.labeler->accu_reward;
        }
        std::ofstream outfile;
        outfile.open("../saved_model/rewards.txt", std::ios_base::app);  
        outfile << engine.searcher.labeler->avg_reward << ", ";  
        outfile.close();
    }

}

void earlyStoppingCallbackImpl(void* engine_ptr)
{
    updateCallbackImpl(engine_ptr);
}

void optimalFoundCallbackImpl(void* engine_ptr)
{
    updateCallbackImpl(engine_ptr);
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
    string qNetPath = QNetPathInf;
    string piNetPath = PiNetPathInf;
    string qOptimPath = QOptimPathInf;
    string piOptimPath = PiOptimPathInf;
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
                .num_epoch(10) 
                .inference_start_epoch(1)
                .entropy_lambda(1)                
                .lr_pi(1e-5*0.3)      
                .lr_q(1e-4*0.3)                
                .steps_per_epoch(10000)
                .buffer_size(5000)
        );
    
    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {              
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;
                      
        string filename(argv[2]);  

        /* Inference! */
        labeler->eval();
        /* ========== */

        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {                
            cerr << "epoch: " << epoch << endl;
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
            labeler->epoch()++;   
        }
    }
    else if (argc >= 3 && !(strcmp(argv[1], "-d")))
    {        
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;
        
        // read problem
        InputHandler inputHandler((string(argv[2])));
        string filepath;

        // create folder for problem
        std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        string create_file_cmd = "mkdir " + std::to_string(now_time);
        exec(create_file_cmd.c_str());

        /* Inference! */
        labeler->eval();
        /* ========== */

        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {               
            filepath = inputHandler.getNextFileName();                                            
            if(filepath.empty())
            {
                inputHandler.reset(); 
                filepath = inputHandler.getNextFileName();                                            
            }            

            if(parse_and_init_oneRjSumCj(filepath))
            {
                cerr << "epoch: " << epoch << " / " << labeler->opt.num_epoch() << " : " << filepath << endl;
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
            

                // compare with plain CBFS
                string plain_cmd = "./" + string(PlainCBFSVerbose) + " -f " + filepath;
                cout << "executing \"" << plain_cmd << "\"" << endl;
                string plain_cmd_out = exec(plain_cmd.c_str());
                cout << plain_cmd_out << endl; // redirect the output to stdout
                cout << "execution ended" << endl;     
                
                // different command for different strategies
                string plain_bfs_cmd = "./" + string(PlainCBFSBFS) + " -f " + filepath;
                string plain_level_cmd = "./" + string(PlainCBFSLevel) + " -f " + filepath;
                string plain_rand_cmd = "./" + string(PlainCBFSRand) + " -f " + filepath;

                // print the result to file
                std::ofstream outfile;
                string filename = "./" + std::to_string(now_time) + "/" + filepath.substr(filepath.find_last_of("/") + 1);;
                cerr << "writing in file: " << filename << endl;
                outfile.open(filename, std::ios_base::app);    
                outfile << graph.searched_node_num << endl; // 1
                outfile << exec(plain_bfs_cmd.c_str()); // 2
                outfile << exec(plain_level_cmd.c_str()); // 3
                outfile << exec(plain_rand_cmd.c_str()); // 4
                outfile.close();
            }

            labeler->epoch()++;   
        }         
    }
}

