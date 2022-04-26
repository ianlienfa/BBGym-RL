#ifndef DDPRLABELER_H
#define DDPRLABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"
#include "search_modules/Net/DDPR/NetDDPR.h"

typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Batch;
typedef std::tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> RawBatch;
typedef std::tuple<float, float, float> ActorOut;

struct DDPRLabelerOptions{
    float gamma;
    float lr_q;
    float lr_pi;
    float polyak;
    int64_t num_epoch;
    int64_t max_steps;
    int64_t update_start_epoch;
    int64_t buffer_size;
    float noise_scale;
    float epsilon ;
    int64_t batch_size;
    int64_t update_freq;
    int64_t tail_updates;
    int operator_option;
    int64_t max_num_contour;
    DDPRLabelerOptions();
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
    int64_t operator_options;
    int64_t tail_updates;
    int64_t max_num_contour;
    
    
    NetDDPR net{nullptr};
    NetDDPR net_tar{nullptr};    
    ReplayBuffer buffer{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_q{nullptr}, optimizer_pi{nullptr};    

    // Random contour map
    vector<int> contour_candidates;

    // Trackers
    float last_action;
    vector<float> test_loss_vec;    
    vector<float> q_loss_vec;
    vector<float> pi_loss_vec;
    vector<float> q_mean_loss;
    vector<float> pi_mean_loss;    
    int64_t step;
    int64_t update_count;
    int64_t epoch;

    // Operator choices
    struct OperatorOptions{
        static constexpr int RANDOM = 0;
        static constexpr int TRAIN = 1;
        static constexpr int INFERENCE = 2;
        static constexpr int TESTING = 3;
    };

    DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, string load_q_path = "", string load_pi_path = "", string q_optim_path = "", string pi_optim_path = "", DDPRLabelerOptions options = DDPRLabelerOptions());    
    void fill_option(const DDPRLabelerOptions &options);
    // float operator()(StateInput input);
    float operator()(vector<float> flatten, int operator_option);
    ActorOut train(vector<float> flatten, int operator_option);
    float label_decision(const ActorOut &in);
    float label_decision(ActorOut &in, bool explore, float epsilon=0.5);
    torch::Tensor compute_q_loss(const Batch &batch_data);
    torch::Tensor compute_pi_loss(const Batch &batch_data);
    void update(const RawBatch &batch_data);
    float get_action(const StateInput &input);
};




#endif