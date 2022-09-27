#ifndef PPOLABELER_H
#define PPOLABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/PPO/NetPPO.h"
#include <cmath>
namespace PPO
{
struct PPOLabelerOptions{
    // network structure parameters
    BBARG(PPOLabelerOptions, int64_t, state_dim, 0);
    BBARG(PPOLabelerOptions, int64_t, action_dim, 0);
    BBARG(PPOLabelerOptions, int64_t, hidden_dim, 64);

    // hyper parameters
    BBARG(PPOLabelerOptions, int64_t, num_epoch, 300);
    BBARG(PPOLabelerOptions, int64_t, epoch_per_instance, 10);
    BBARG(PPOLabelerOptions, int64_t, epochs_per_update, 5); // sit it bigger to provide value fitting a better estimate
    BBARG(PPOLabelerOptions, int64_t, inference_start_epoch, INT_MAX);
    BBARG(PPOLabelerOptions, int64_t, validation_interval, 5);  // instances per validation
    BBARG(PPOLabelerOptions, int64_t, steps_per_epoch, 4000);
    BBARG(PPOLabelerOptions, int64_t, train_pi_iter, 80);
    BBARG(PPOLabelerOptions, int64_t, train_q_iter, 80);
    BBARG(PPOLabelerOptions, float, gamma, 0.99);
    BBARG(PPOLabelerOptions, float, entropy_lambda, 0.1);
    BBARG(PPOLabelerOptions, float, lr_q, 1e-6);
    BBARG(PPOLabelerOptions, float, lr_pi, 1e-7);
    BBARG(PPOLabelerOptions, float, clip_ratio, 0.2);
    BBARG(PPOLabelerOptions, float, target_kl, 0.001);

    // Not determined
    BBARG(PPOLabelerOptions, string, load_q_path, "");
    BBARG(PPOLabelerOptions, string, load_pi_path, "");
    BBARG(PPOLabelerOptions, string, q_optim_path, "");
    BBARG(PPOLabelerOptions, string, pi_optim_path, "");

    BBARG(PPOLabelerOptions, int64_t, buffer_size, 5000);
    BBARG(PPOLabelerOptions, int64_t, max_num_contour, 100);
};

struct PPOLabeler: Labeler
{ 
public:
    PPO::PPOLabelerOptions opt;
    NetPPO net{nullptr};    
    PPO::ReplayBuffer buffer{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_q{nullptr}, optimizer_pi{nullptr};        

    // global trackers
    vector<float> q_loss_vec;
    vector<float> pi_loss_vec;
    vector<float> q_mean_loss;
    vector<float> pi_mean_loss;    
    vector<float> ewma_reward_vec;
    vector<float> avg_search_decrease_ratio_vec;
    vector<float> ewma_search_decrease_ratio_vec;
    float avg_search_decrease_ratio = 0;
    float accu_reward = 0;
    float avg_reward = 0;
    float avg_inf_reward = 0;    
    float last_searched_node_num = 0.0;    
    int64_t per_instance_epoch = 0;
    const float * real_rewards;
    const int64_t * step;
  
    // Trackers tracks training state
    enum LabelerState {UNDEFINED, TRAIN_RUNNING, TRAIN_EPOCH_END, INFERENCE, TESTING} labeler_state;    
    BBARG(PPOLabeler, int64_t, update_count, 0);
    BB_FUNC_ARG(PPOLabeler, int64_t, epoch, 0, [&](int64_t value){this->per_instance_epoch += value; cout << "per instance epoch: " << this->per_instance_epoch << endl;});

    // contour status tracker
    BBARG(PPOLabeler, int64_t, current_contour_pointer, 0);    
    std::unique_ptr<vector<int>> contour_config_ptr = nullptr;

    // network training helpers        
    PPOLabeler(PPO::PPOLabelerOptions options = PPO::PPOLabelerOptions());        
    int64_t operator()(::OneRjSumCjNode& node, vector<float>& flatten, ::OneRjSumCjGraph& graph); //returns position
    void train(){labeler_state = LabelerState::TRAIN_RUNNING;}
    void eval(){labeler_state = LabelerState::INFERENCE;}
    LabelerState get_labeler_state();
    // STATE_ENCODING get_state();
    tuple<torch::Tensor, PPO::ExtraInfo> compute_pi_loss(const PPO::Batch &batch_data);
    torch::Tensor compute_q_loss(const PPO::Batch &batch_data);
    void reset_instance_trackers();
    void update(PPO::SampleBatch &batch_data); 
    torch::Tensor step_based_entropy_loss_decorator(torch::Tensor loss, float entropy, float epoch);
    };

}; //namespace: PPO

#endif