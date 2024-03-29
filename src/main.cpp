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
    {
        qNetPath = "";
        cerr << "qNet model for evaluation not exist" << endl;
        exit(1);
    }

    if(!std::filesystem::exists(piNetPath))
    {
        piNetPath = "";
        cerr << "qNet model for evaluation not exist" << endl;
        exit(1);
    }
    if(!std::filesystem::exists(qOptimPath))
        qOptimPath = "";
    if(!std::filesystem::exists(piOptimPath))
        piOptimPath = "";    
    
    const int64_t max_num_contour = V_MAX_NUM_CNTR;

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
                .num_epoch(50) 
                .inference_start_epoch(1)
                .entropy_lambda(1)                
                .lr_pi(1e-5*0.3)      
                .lr_q(1e-4*0.3)                
                .steps_per_epoch(10000)
                .buffer_size(5000)
        );
    
    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {              
        vector<int> strategy_searched_nodes;
        vector<int> strategy_won = vector<int>(4);
        float total_trials = 0;
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;
                      
        string filepath(argv[2]);  

        /* Inference! */
        labeler->eval();
        /* ========== */

        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {                
            cerr << "epoch: " << epoch << endl;
            if(parse_and_init_oneRjSumCj(filepath))
            {
                OneRjSumCjSearch searcher(labeler);
                OneRjSumCjBranch brancher;
                OneRjSumCjPrune pruner;
                LowerBound lowerbound;
                OneRjSumCjGraph graph;
                OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                graph = solver.solve(OneRjSumCjNode());                  
            
                // different command for different strategies
                string plain_bfs_cmd = "./" + string(PlainCBFSBFS) + " -f " + filepath;
                string plain_level_cmd = "./" + string(PlainCBFSLevel) + " -f " + filepath;
                string plain_rand_cmd = "./" + string(PlainCBFSRand) + " -f " + filepath;

                // print the result to file
                std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::ofstream outfile;
                string filename = "./" + std::to_string(now_time) + "." + filepath.substr(filepath.find_last_of("/") + 1);;
                cerr << "writing in file: " << filename << endl;
                outfile.open(filename, std::ios_base::app);   
                strategy_searched_nodes[0] = graph.searched_node_num;
                strategy_searched_nodes[1] = stoi(exec(plain_bfs_cmd.c_str()));
                strategy_searched_nodes[2] = stoi(exec(plain_level_cmd.c_str()));
                strategy_searched_nodes[3] = stoi(exec(plain_rand_cmd.c_str()));
                outfile << graph.searched_node_num << endl; // 1
                outfile << strategy_searched_nodes[1]; // 2
                outfile << strategy_searched_nodes[2]; // 3
                outfile << strategy_searched_nodes[3]; // 4
                strategy_won[std::min_element(strategy_searched_nodes.begin(), strategy_searched_nodes.end()) - strategy_searched_nodes.begin()] += 1;
                total_trials++;
                outfile.close();
            }
            labeler->epoch(epoch); 
        }
        auto percentage = [](int val, int total_trials) { return (float)val / total_trials; };
        cerr << "----- summary: -----" << endl;
        cerr << "net: " << percentage(strategy_won[0], total_trials) << "%" << endl;
        cerr << "bfs: " << percentage(strategy_won[1], total_trials) << "%" << endl;
        cerr << "level: " << percentage(strategy_won[2], total_trials) << "%" << endl;
        cerr << "rand: " << percentage(strategy_won[3], total_trials) << "%" << endl;
    }
    else if (argc >= 3 && !(strcmp(argv[1], "-d")))
    {        
        vector<int> strategy_searched_nodes = vector<int>(4);
        vector<int> strategy_won = vector<int>(4);
        vector<float> best_worst_gaps;
        float total_trials = 0;
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
            

                // // compare with plain CBFS
                // string plain_cmd = "./" + string(PlainCBFSVerbose) + " -f " + filepath;
                // cout << "executing \"" << plain_cmd << "\"" << endl;
                // string plain_cmd_out = exec(plain_cmd.c_str());
                // cout << plain_cmd_out << endl; // redirect the output to stdout
                // cout << "execution ended" << endl;     
                
                // different command for different strategies
                string plain_bfs_cmd = "./" + string(PlainCBFSBFS) + " -f " + filepath;
                string plain_level_cmd = "./" + string(PlainCBFSLevel) + " -f " + filepath;
                string plain_rand_cmd = "./" + string(PlainCBFSRand) + " -f " + filepath;

                // print the result to file
                std::ofstream outfile;
                string filename = "./" + std::to_string(now_time) + "/" + filepath.substr(filepath.find_last_of("/") + 1);;
                cerr << "writing in file: " << filename << endl;
                outfile.open(filename, std::ios_base::app);   
                strategy_searched_nodes[0] = graph.searched_node_num;
                strategy_searched_nodes[1] = stoi(exec(plain_bfs_cmd.c_str()));
                strategy_searched_nodes[2] = stoi(exec(plain_level_cmd.c_str()));
                strategy_searched_nodes[3] = stoi(exec(plain_rand_cmd.c_str()));
                outfile << graph.searched_node_num << endl; // 1
                outfile << strategy_searched_nodes[1] << endl; // 2
                outfile << strategy_searched_nodes[2] << endl; // 3
                outfile << strategy_searched_nodes[3]<< endl; // 4
                outfile.close();
                strategy_won[std::min_element(strategy_searched_nodes.begin(), strategy_searched_nodes.end()) - strategy_searched_nodes.begin()] += 1;
                float largest = *std::max_element(strategy_searched_nodes.begin(), strategy_searched_nodes.end());
                float smallest = *std::min_element(strategy_searched_nodes.begin(), strategy_searched_nodes.end());

                // only compute the comparison of net and worst strategy                
                best_worst_gaps.push_back((largest - strategy_searched_nodes[0]) / largest);
                total_trials++;
            }

            labeler->epoch(epoch);
        }         
        auto percentage = [](int val, int total_trials) { return (float)val / total_trials; };
        cerr << "----- summary: -----" << endl;
        cerr << std::setprecision(3);
        cerr << "net: " << percentage(strategy_won[0], total_trials) << "%" << endl;
        cerr << "bfs: " << percentage(strategy_won[1], total_trials) << "%" << endl;
        cerr << "level: " << percentage(strategy_won[2], total_trials) << "%" << endl;
        cerr << "rand: " << percentage(strategy_won[3], total_trials) << "%" << endl;        
        cerr << best_worst_gaps << endl;
        cerr << "Adjusted best/worst gap: " << std::accumulate(best_worst_gaps.begin(), best_worst_gaps.end(), 0.0) / best_worst_gaps.size() << endl;
    }
}

