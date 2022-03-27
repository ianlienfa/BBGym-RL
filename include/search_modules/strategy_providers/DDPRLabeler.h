#ifndef DDPRLABELER_H
#define DDPRLABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/DDPR/NetDDPR.h"

typedef tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Batch;

struct DDPRLabelerOptions{
    float gamma=0.99;
    float lr_q=1e-3;
    float lr_pi=1e-3;
    float polyak=0.995;
    int64_t num_epoch=150;
    int64_t max_steps=1000;
    int64_t update_start_epoch=10;
    int64_t buffer_size=int64_t(1e6);
    float noise_scale=0.1;
    float epsilon = 0.5;
    int64_t batch_size=100;
    int64_t update_freq=5;
    DDPRLabelerOptions(){};
};

struct DDPRLabeler: Labeler
{    
    int64_t state_dim;
    int64_t action_dim;
    Pdd action_range;

    float  gamma;
    float  lr_q;
    float  lr_pi;
    float  polyak;
    float  noise_scale;
    float  epsilon;
    int64_t num_epoch;
    int64_t max_steps;
    int64_t update_start_epoch;
    int64_t buffer_size;
    int64_t batch_size;
    int64_t update_freq;
    
    NetDDPR net{nullptr};
    NetDDPR net_tar{nullptr};    
    ReplayBuffer buffer{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_q{nullptr}, optimizer_pi{nullptr};    

    // Trackers
    float last_action;
    vector<float> test_loss;
    vector<float> loss_epoch;
    int step;
    int epoch;


    DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, DDPRLabelerOptions options = DDPRLabelerOptions());
    void fill_option(const DDPRLabelerOptions &options);
    // float operator()(StateInput input);
    float operator()(vector<float> flatten, bool is_train = false);
    float ddpg_train(torch::Tensor tensor_in);

    torch::Tensor compute_q_loss(const Batch &batch_data);
    torch::Tensor compute_pi_loss(const Batch &batch_data);
    void update(Batch &batch_data);
    float get_action(const StateInput &input);
};

#endif