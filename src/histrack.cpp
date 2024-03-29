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
    // engine.searcher.labeler->step() = 0;

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

string splitFileName (const std::string& str)
{
  std::size_t found = str.find_last_of("/");
  string file = str.substr(found+1);
  return file;  
}

enum class ProblemType {
    NullType,
    SingleInstanceTrain,
    SingleInstanceInf,
    SingleInstanceBackTrace,
    MultipleInstanceTrain,
    MultipleInstanceValidate,
};

std::map<string, vector<string>> parse_args(int argc, char* argv[])
{
    std::map<string, vector<string>> commands;
    std::map<string, vector<string>>::iterator it = commands.end();
    for(int i = 0; i < argc; i++)
    {
        cout << string(argv[i]) << endl;
        if(string(argv[i])[0] == '-')
        {
            commands.insert(make_pair(string(argv[i]), vector<string>()));            
            it = commands.find(string(argv[i]));
        }
        else
        {
            if(it != commands.end())
            {                           
                it->second.push_back(string(argv[i]));
            }
        }
    }    
    return commands;
}


struct SolverOptions
{
    string path = "";
    string filename = "";
    ProblemType problem_type = ProblemType::NullType;
    int64_t trained_epoch;
    bool average = false;
    int64_t avg_quant = 10;  
    int64_t start_epoch = 1000;      
};

struct AvgCounter
{
    int quant = 0;    
    int count = 0;
    float accum = 0;
    AvgCounter(int64_t quant): quant(quant) {}
    void add(float val)
    {        
        cout << "adding " << val << " to avg counter: " << count << " / " << quant << endl;
        count++;
        accum += val;
    }
    bool is_full()
    {
        return count == quant;
    }
    float_t get_avg_and_reset()
    {
        if(count < quant)
        {
            cout << "Warning: avg counter is not full" << endl;
            exit(1);
        }
        else
        {            
            float avg = accum / quant;
            count = 0;
            accum = 0;
            return avg;
        }
    }
};


SolverOptions getSolverOptions(int argc, char* argv[])
{
    SolverOptions options;
    auto commands = parse_args(argc, argv);
    for(auto &command : commands)
    {
        if(command.first == "-avg")
        {
            options.average = true;
            if(command.second.size() == 1)
            {
                auto is_num = [](const std::string &s) {
                    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
                };
                if(is_num(command.second[0]))
                {
                    options.avg_quant = stoi(command.second[0]);
                }
            }
        }
        if(command.first == "-st")
        {            
            if(command.second.size() == 1)
            {
                auto is_num = [](const std::string &s) {
                    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
                };
                if(is_num(command.second[0]))
                {
                    options.start_epoch = stoi(command.second[0]);
                }
            }
        }
        if(command.first == "-f")
        {
            if(command.second.size() != 1)
            {
                cout << "Usage: ./net_ddpr -f <problem_file>" << endl;
                exit(1);
            }
            options.path = command.second[0];
            options.problem_type = ProblemType::SingleInstanceTrain;
        }
        else if(command.first == "-bt")
        {
            // -bt <trained_epoch> <dir_path> <problem_file>
            cout << command.second << endl;
            cout << "command.second.size() = " << command.second.size() << endl;
            if(command.second.size() != 3)
            {
                cout << "Usage: ./executable -bt <trained_epoch> <dir_path> <problem_file>" << endl;
                exit(1);
            }                        
            options.trained_epoch = int64_t(stoi(command.second[0]));            
            options.path = command.second[1];
            options.filename = command.second[2];
            options.problem_type = ProblemType::SingleInstanceBackTrace;            
        } 
        else if(command.first == "-mulval")
        {
            // -mulval <trained_epoch> <dir_path> <validation_instance_path>
            cout << command.second << endl;
            cout << "command.second.size() = " << command.second.size() << endl;
            if(command.second.size() != 3)
            {
                cout << "Usage: ./executable -bt <trained_epoch> <dir_path> <validation_instance_path>" << endl;
                exit(1);
            }                        
            options.trained_epoch = int64_t(stoi(command.second[0]));            
            options.path = command.second[1];
            options.filename = command.second[2];
            options.problem_type = ProblemType::MultipleInstanceValidate;            
        } 
        else
        {
            cout << "not supported command: '" << command.first << "'" << endl;
        }
    }    
    return options;
}



int main(int argc, char* argv[])
{                
    SolverOptions options = getSolverOptions(argc, argv);
    AvgCounter* avg_counter = nullptr;
    if(options.average)
    {
        avg_counter = new AvgCounter(options.avg_quant);
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
                .num_epoch(20) 
                .inference_start_epoch(1)
                .entropy_lambda(1)                
                .lr_pi(V_LR_PI)      
                .lr_q(V_LR_Q)                
                .steps_per_epoch(10000)
                .buffer_size(5000)
        );

    int rand_seed = RANDOM_SEED;
    torch::manual_seed(RANDOM_SEED);    
    cerr << "Random seed: " << rand_seed << endl;
    
    if(options.problem_type == ProblemType::SingleInstanceBackTrace)
    {
        cerr << "SingleInstanceBackTrace start" << endl;
        string epoch_postfix = "";
        const string piNetPathPrefix = options.path + "/piNet_";
        const string qNetPathPrefix = options.path + "/qNet_";
        int64_t num_epoch = options.start_epoch;        
        string outfilename = "./" + splitFileName(options.filename) + ".bt";
        string rm_command = "rm " + outfilename;
        exec(rm_command.c_str());
        do
        {
            epoch_postfix = to_string(num_epoch) + ".pt";
            std::shared_ptr<PPO::PPOLabeler> labeler = 
            std::make_shared<PPO::PPOLabeler>(
                PPO::PPOLabelerOptions()                
                    .state_dim(int64_t(PPO::StateInput(OneRjSumCjNode(), OneRjSumCjNode(), OneRjSumCjGraph().set_max_size(max_num_contour)).get_state_encoding(max_num_contour).size()))
                    .action_dim(4)
                    .load_q_path(qNetPathPrefix + epoch_postfix)
                    .load_pi_path(piNetPathPrefix + epoch_postfix)
                    .q_optim_path("")
                    .pi_optim_path("")
                    .max_num_contour(max_num_contour)                             
                    .buffer_size(5000)
            );

            /* Inference! */
            labeler->eval();
            /* ========== */

            if(parse_and_init_oneRjSumCj(options.filename))
            {
                OneRjSumCjSearch searcher(labeler);
                OneRjSumCjBranch brancher;
                OneRjSumCjPrune pruner;
                LowerBound lowerbound;
                OneRjSumCjGraph graph;
                OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                graph = solver.solve(OneRjSumCjNode());                  
            
                if(options.average)
                {
                    avg_counter->add(graph.searched_node_num);
                    if(avg_counter->is_full())
                    {
                        std::ofstream outfile;
                        cerr << "writing in file: " << outfilename << endl;
                        outfile.open(outfilename, std::ios_base::app);    
                        outfile << avg_counter->get_avg_and_reset() << endl;
                        outfile.close();
                    }
                }
                else
                {
                    std::ofstream outfile;
                    cerr << "writing in file: " << outfilename << endl;
                    outfile.open(outfilename, std::ios_base::app);    
                    outfile << graph.searched_node_num << endl; // 1
                    outfile.close();
                }
            }

            num_epoch += 100;

        } while (num_epoch <= options.trained_epoch);
    }
    else if(options.problem_type == ProblemType::MultipleInstanceValidate)
    {
        cerr << "MultipleInstanceValidate start" << endl;
        string epoch_postfix = "";
        const string piNetPathPrefix = options.path + "/piNet_";
        const string qNetPathPrefix = options.path + "/qNet_";
        std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());        
        string outfilename = "./" + splitFileName(options.filename) + "_"+ std::to_string(now_time) + ".bt";
        InputHandler inputHandler(options.filename);

        cout << "trained_epoch: " << options.trained_epoch << endl;
        for(int64_t num_epoch = options.start_epoch; num_epoch <= options.trained_epoch; num_epoch += 200)
        {
            epoch_postfix = to_string(num_epoch) + ".pt";
            cerr << "attempt to read: " << piNetPathPrefix + epoch_postfix << ", " << qNetPathPrefix + epoch_postfix << endl;
            if(!std::filesystem::exists(qNetPathPrefix + epoch_postfix))
            {
                continue;
            }
            cerr << "epoch_postfix: " << epoch_postfix << endl;
            std::shared_ptr<PPO::PPOLabeler> labeler = 
            std::make_shared<PPO::PPOLabeler>(
                PPO::PPOLabelerOptions()                
                    .state_dim(int64_t(PPO::StateInput(OneRjSumCjNode(), OneRjSumCjNode(), OneRjSumCjGraph().set_max_size(max_num_contour)).get_state_encoding(max_num_contour).size()))
                    .action_dim(4)
                    .load_q_path(qNetPathPrefix + epoch_postfix)
                    .load_pi_path(piNetPathPrefix + epoch_postfix)
                    .q_optim_path("")
                    .pi_optim_path("")
                    .max_num_contour(max_num_contour)                             
                    .buffer_size(5000)
            );

            /* Inference! */
            labeler->eval();
            /* ========== */

            float total_searched_node_num = 0;
            float total_visited_instance_num = 0;
            string next_file = inputHandler.getNextFileName();
            cout << "file list: " << inputHandler.file_list << endl;
            while(!inputHandler.isEnd())
            {   
                next_file = inputHandler.getCurrentFileName();
                if(!std::filesystem::is_regular_file(next_file))
                    break;
                cout << "next_file: " << next_file << endl;
                if(parse_and_init_oneRjSumCj(next_file))
                {
                    OneRjSumCjSearch searcher(labeler);
                    OneRjSumCjBranch brancher;
                    OneRjSumCjPrune pruner;
                    LowerBound lowerbound;
                    OneRjSumCjGraph graph;
                    OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                    graph = solver.solve(OneRjSumCjNode());                                  
                    total_searched_node_num += graph.searched_node_num;
                    total_visited_instance_num += 1;
                }
                inputHandler.toNextFileWithEnd();
            }
            if(options.average)
            {
                avg_counter->add(total_searched_node_num / total_visited_instance_num);
                if(avg_counter->is_full())
                {
                    std::ofstream outfile;
                    cerr << "writing in file: " << outfilename << endl;
                    outfile.open(outfilename, std::ios_base::app);    
                    outfile << avg_counter->get_avg_and_reset() << endl;
                    outfile.close();
                }
            }
            else
            {
                std::ofstream outfile;
                cerr << "writing in file: " << outfilename << endl;
                outfile.open(outfilename, std::ios_base::app);   
                cout << "total_searched_node_num: " << total_searched_node_num << "total_visited_instance_num: " << total_visited_instance_num << endl;
                outfile << epoch_postfix << " : " << total_searched_node_num / total_visited_instance_num << endl; // 1
                outfile.close();
            }
        
            inputHandler.reset();
            // num_epoch += 100;

        } 
    }
    else
    {
        cerr << "problem type is not defined" << endl;
    }

}

