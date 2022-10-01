#include <iostream>
#include <stdexcept>
#include <memory>
#include <unordered_map>
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
    if(current_labeler_state == PPO::PPOLabeler::LabelerState::TRAIN_RUNNING || current_labeler_state == PPO::PPOLabeler::LabelerState::TRAIN_EPOCH_END)
    {
        PPO::SampleBatch batch = labeler->buffer->get();
        if(batch.v_r.size() > 0)
        {
            labeler->update(batch);
        }        
        else
        {
            std::cout << "batch size is too small, skip update" << std::endl;
            exit(1);
        }
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
    // trainer global variables    
    string filename = "";
    string validation_dirname = "";

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
            else if(command.first == "-d")
            {   
                validation_dirname = command.second[0];
            }
            else if(command.first == "-param")
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
                .num_epoch(6000)
                .inference_start_epoch(1000)
                .epoch_per_instance(10)
                .epochs_per_update(1)
                .validation_interval(20)
                .entropy_lambda(0.3)                
                .lr_pi(V_LR_PI)      
                .lr_q(V_LR_Q)                
                .steps_per_epoch(100000)             
                .buffer_size(150000)
                .target_kl(0.00001)
                .train_pi_iter(50)
                .train_q_iter(50)
        );
    
    /* validate */
    auto validate = [&](string dirpath){

        labeler->eval();
        
        // Input handeler
        InputHandler inputHandler_test(dirpath);
        // count number of files
        int num_valid_files = stoi(exec(string("ls " + dirpath +  " | wc -l").c_str()));

        cerr << "validating : " << labeler->epoch() << " / " << labeler->opt.num_epoch() << endl;

        // initialize if not initialized before;
        static std::unordered_map<string, int> base_searchnum_map;
        if(!base_searchnum_map.size())
        {
            InputHandler base_searchnum_initializer(dirpath);
            for(int file = 0; file <= num_valid_files; file++)
            {
                string filepath = base_searchnum_initializer.getNextFileName();                 
                string plain_bfs_cmd = "./" + string(PlainCBFSBFS) + " -f " + filepath;
                int search_num = stoi(exec(plain_bfs_cmd.c_str()));
                base_searchnum_map.insert(make_pair(filepath, search_num));
            }
        }

        for(int epoch = 1; epoch <= num_valid_files; epoch++)      
        {
            string filepath = inputHandler_test.getNextFileName();  
            if(parse_and_init_oneRjSumCj(filepath))
            {
                cout <<  "validating on: " << filepath << endl;            
                OneRjSumCjSearch searcher(labeler);
                OneRjSumCjBranch brancher;
                OneRjSumCjPrune pruner;
                LowerBound lowerbound;
                OneRjSumCjGraph graph;
                OneRjSumCj_engine solver(graph, searcher, brancher, pruner, lowerbound); 
                graph = solver.solve(OneRjSumCjNode());  

                // Create file if not exist 
                if(!std::filesystem::exists("../saved_model/ratio.txt"))
                {
                    std::ofstream outfile("../saved_model/ratio.txt");
                    outfile.close();
                }          

                // clean up loss_vec (leave this to prevent validation loss polluting the vectors)
                labeler->q_loss_vec.clear();
                labeler->pi_loss_vec.clear();

                // get corresponding bfs search (base) node search num
                auto base_node_search_num_ptr = base_searchnum_map.find(filepath);
                int base_node_search_num;
                if(base_node_search_num_ptr == base_searchnum_map.end())
                {
                    string msg = "base number of node search is not filled for " + filepath; 
                    assertm(msg, false);
                }
                else
                {
                    base_node_search_num = base_node_search_num_ptr->second;
                }

                // track search node saved ratio                 
                float node_saved_ratio = float(base_node_search_num - graph.searched_node_num) / float(base_node_search_num);
                cout << filepath << " | base: " << base_node_search_num << ", trained: " << graph.searched_node_num << ", node-saved-ratio: " << node_saved_ratio << endl;
                labeler->ewma_search_decrease_ratio_vec.push_back(node_saved_ratio);            
            }
        }        

        // save average performance into ratio.txt
        float search_decrease_ratio = std::accumulate(labeler->ewma_search_decrease_ratio_vec.begin(), labeler->ewma_search_decrease_ratio_vec.end(), 0.0) / labeler->ewma_search_decrease_ratio_vec.size();                
        labeler->avg_search_decrease_ratio = search_decrease_ratio * 0.9 + labeler->avg_search_decrease_ratio * 0.1;
        labeler->ewma_search_decrease_ratio_vec.clear();
        cerr << "avg_search_decrease_ratio: " << labeler->avg_search_decrease_ratio << endl;
        if(!std::filesystem::exists("../saved_model/ratio.txt"))
        {
            std::ofstream outfile("../saved_model/ratio.txt");
            outfile.close();
        }
        std::ofstream outfile;
        outfile.open("../saved_model/ratio.txt", std::ios_base::app);    
        outfile << labeler->avg_search_decrease_ratio << ", ";
        outfile.close();
        labeler->avg_search_decrease_ratio_vec.push_back(labeler->avg_search_decrease_ratio);


        // toggle the training mode on to leave the validation process
        labeler->train();
    };

    if (argc >= 3 && !(strcmp(argv[1], "-f")))
    {              
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;                    
        string filename(argv[2]);  
        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {                
            if(epoch == labeler->opt.num_epoch())
            {
                labeler->eval();
            }
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
                // if(epoch % 500 == 0)
                // {
                //     torch::save(labeler->net->q, "../saved_model/qNet_" + std::to_string(epoch) + ".pt");
                //     torch::save(labeler->net->pi, "../saved_model/piNet_" + std::to_string(epoch) + ".pt");
                //     torch::save((*labeler->optimizer_q), "../saved_model/optimizer_q_" + std::to_string(epoch) + ".pt");
                //     torch::save((*labeler->optimizer_pi), "../saved_model/optimizer_pi_" + std::to_string(epoch) + ".pt"); 
                // }                
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

                // stage save
                if(epoch % 100 == 0)
                {
                    string cmd = "cp ../saved_model/qNet.pt ../saved_model/qNet_" + std::to_string(epoch) + ".pt";
                    exec(cmd.c_str());
                    cmd = "cp ../saved_model/piNet.pt ../saved_model/piNet_" + std::to_string(epoch) + ".pt";
                    exec(cmd.c_str());
                }

                cerr << "epoch: " << epoch << " / " << labeler->opt.num_epoch() << ", buffer: " << labeler->buffer->start_idx << "/" << labeler->opt.buffer_size() << endl;
                labeler->epoch(1);
            }

            #if INF_MODE == 1
            epoch = labeler->num_epoch;
            #endif
        }
    }
    else if (argc >= 3 && !(strcmp(argv[1], "-d")))
    {        
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;

        // count number of files
        // int num_files = stoi(exec(string("ls " + string(argv[2]) +  " | wc -l").c_str()));
        
        // read problem
        InputHandler inputHandler((string(argv[2])));
        InputHandler inputHandler_test((string(argv[2])) + "/validation");
        string validation_filepath = string((string(argv[2])) + "/validation");
        string filepath = "";

        for(int epoch = 1; epoch <= labeler->opt.num_epoch(); epoch++)              
        {                
            if((epoch % labeler->opt.epoch_per_instance() == 0 || epoch == 1))
            {        
                labeler->per_instance_epoch = 0;                      
                cerr << "training.." << endl;
                labeler->train();
                // int step_size = 1;
                // for(int i = 0; i < step_size; i++)
                // {
                filepath = inputHandler.getNextFileName();  
                cerr << filepath << endl;

                // }
            }

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
                if(epoch % 100 == 0)
                {
                    torch::save(labeler->net->q, "../saved_model/qNet_" + std::to_string(epoch) + ".pt");
                    torch::save(labeler->net->pi, "../saved_model/piNet_" + std::to_string(epoch) + ".pt");
                    torch::save((*labeler->optimizer_q), "../saved_model/optimizer_q_" + std::to_string(epoch) + ".pt");
                    torch::save((*labeler->optimizer_pi), "../saved_model/optimizer_pi_" + std::to_string(epoch) + ".pt"); 
                }
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

                cerr << "epoch: " << epoch << " / " << labeler->opt.num_epoch() << " : " << filepath << ", buffer: " << labeler->buffer->start_idx << "/" << labeler->opt.buffer_size() << endl;
    
                // place this before validation to sync the epoch
                labeler->epoch(1);   
            }
            
            // validation at each validation interval
            // if(epoch % (labeler->opt.validation_interval() * labeler->opt.epoch_per_instance()) == 0 && epoch > labeler->opt.inference_start_epoch())
            // {
            //     validate(validation_filepath);
            // }  
        }       
    }
    else if (argc >= 3 && !(strcmp(argv[1], "-i")))
    {        
        int rand_seed = RANDOM_SEED;
        torch::manual_seed(RANDOM_SEED);    
        cerr << "Random seed: " << rand_seed << endl;                    
        string filename(argv[2]);  
        
        // inference
        labeler->eval();
        
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
