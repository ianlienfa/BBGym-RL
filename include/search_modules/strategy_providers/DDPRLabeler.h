#ifndef PLAIN_LABELER_H
#define PLAIN_LABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/DDPR/NetDDPR.h"

typedef tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Batch;

struct DDPRLabeler: Labeler
{    
    double gamma;
    double lr_q;
    double lr_pi;
    double polyak;
    double noise_scale;
    int64_t num_epoch;
    int64_t max_steps;
    int64_t update_start_epoch;
    int64_t buffer_size;
    int64_t batch_size;
    int64_t update_freq;
    
    NetDDPR net{null};
    NetDDPR net_tar{null};    
    ReplayBuffer buffer;
    torch::optim::Optimizer optimizer_q, optimizer_pi;    

    // Trackers
    vector<float> test_loss;
    vector<float> epoch;
    int step;
    int epoch;

    DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, double gamma=0.99, double lr_q=1e-3, double lr_pi=1e-3, double polyak=0.995, int64_t num_epoch=150, int64_t max_steps=1000, int64_t update_start_epoch=10, int64_t buffer_size=int64_t(1e6), double noise_scale=0.1, int64_t batch_size=100, int64_t update_freq=5);
    void fill_response(float reward, bool done);
    int operator()(const OneRjSumCjNode &node) const;
    int operator()(const StateInput &input) const;
    float ddpg(torch::Tensor tensor_in);

    torch::Tensor compute_q_loss(Batch batch_data);
    torch::Tensor compute_pi_loss(Batch batch_data);
    void update(Batch batch_data);
    float get_action(const StateInput &input);
};

#endif