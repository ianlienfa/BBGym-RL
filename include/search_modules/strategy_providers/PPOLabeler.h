#ifndef PPOLABELER_H
#define PPOLABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/PPO/NetPPO.h"
namespace PPO
{

struct PPOLabelerOptions{

    // network structure parameters
    BBARG(int64_t, state_dim, 0);
    BBARG(int64_t, action_dim, 0);
    BBARG(int64_t, hidden_dim, 64);

    // hyper parameters
    BBARG(int64_t, num_epoch, 300);
    BBARG(int64_t, steps_per_epoch, 4000);
    BBARG(int64_t, train_pi_iter, 80);
    BBARG(int64_t, train_q_iter, 80);
    BBARG(float, gamma, 0.99);
    // BBARG(float, polyak);
    BBARG(float, lr_q, 1e-3);
    BBARG(float, lr_pi, 1e-3);
    BBARG(float, clip_ratio, 0.2);
    BBARG(float, target_kl, 0.01);

    // Not determined
    BBARG(string, load_q_path, "");
    BBARG(string, load_pi_path, "");
    BBARG(string, q_optim_path, "");
    BBARG(string, pi_optim_path, "");

    BBARG(int64_t, steps_per_epoch, 4000);
    BBARG(int64_t, buffer_size, 4000);
};

struct PPOLabeler: Labeler
{ 
    PPOLabelerOptions opt;
    NetPPO net{nullptr};    
    PPO::ReplayBuffer buffer{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_q{nullptr}, optimizer_pi{nullptr};    

    // global trackers
    vector<float> q_loss_vec;
    vector<float> pi_loss_vec;
    vector<float> q_mean_loss;
    vector<float> pi_mean_loss;    
    vector<float> ewma_reward_vec;
  
    // Trackers tracks training state
    enum LabelerState {TRAIN_RUNNING, TRAIN_EPOCH_END, INFERENCE, TESTING} labeler_state;    
    BBARG(int64_t, step, 0);
    BBARG(int64_t, update_count, 0);
    BBARG(int64_t, epoch, 0);
        
    // network training helpers
    PPOLabeler(PPOLabelerOptions options = PPOLabelerOptions());        
    float operator()(vector<float> flatten);
    void train(){labeler_state = LabelerState::TRAIN_RUNNING;}
    void eval(){labeler_state = LabelerState::INFERENCE;}
    LabelerState get_labeler_state();
    tuple<torch::Tensor, PPO::ExtraInfo> compute_pi_loss(const PPO::Batch &batch_data);
    torch::Tensor compute_q_loss(const PPO::Batch &batch_data);
    void update(const PPO::SampleBatch &batch_data);    
};

    // env wrapper: input state and labeler that performs the 4-head gaming-like action and output label at the end
    float env_wrapper(OneRjSumCjGraph &graph, PPOLabeler &labeler, vector<float> state_flat);

}; //namespace: PPO

#endif