#ifndef PPOLABELER_H
#define PPOLABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/PPO/NetPPO.h"
namespace PPO
{

struct PPOLabelerOptions{
    BBARG(int64_t, num_epoch);
    BBARG(int64_t, steps_per_epoch);
    BBARG(float, gamma);
    BBARG(float, polyak);
    BBARG(float, lr_q);
    BBARG(float, lr_pi);
    BBARG(float, clip_ratio);
    BBARG(float, hidden_dim);
    BBARG(float, train_pi_iter);
    BBARG(float, train_q_iter);
    BBARG(float, target_kl);

    // Not determined
    BBARG(string, load_q_path);
    BBARG(string, load_pi_path);
    BBARG(string, q_optim_path);
    BBARG(string, pi_optim_path);

    BBARG(int64_t, max_steps);
    BBARG(int64_t, buffer_size);
    BBARG(int64_t, tail_updates);
    BBARG(int, operator_option);
    BBARG(int64_t, max_num_contour);
    PPOLabelerOptions();
};

struct PPOLabeler: Labeler
{ 
    PPOLabelerOptions opt;
    NetPPO net{nullptr};
    NetPPO net_tar{nullptr};    
    PPO::ReplayBuffer buffer{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_q{nullptr}, optimizer_pi{nullptr};    

    // Trackers tracks training state
    enum LabelerState {RANDOM, TRAIN, INFERENCE, TESTING};
    float last_action;    
    vector<float> q_loss_vec;
    vector<float> pi_loss_vec;
    vector<float> q_mean_loss;
    vector<float> pi_mean_loss;    
    vector<float> ewma_reward_vec;
    int64_t step;
    int64_t update_count;
    int64_t epoch;    
    
    PPOLabeler(int64_t state_dim, int64_t action_dim, PPOLabelerOptions options = PPOLabelerOptions());        
    float operator()(vector<float> flatten, int operator_option);
    void train();
    void eval();
    LabelerState get_labeler_state();

    torch::Tensor compute_q_loss(const PPO::SampleBatch &batch_data);
    torch::Tensor compute_pi_loss(const PPO::SampleBatch &batch_data);
    void update(const PPO::SampleBatch &batch_data);    

    // vec_argmax is 1-based index, for contour_candidates
    float vec_argmax(const vector<float> &v){return std::distance(v.begin(), std::max_element(v.begin(), v.end())) + 1;}
};

}; //namespace: PPO

#endif