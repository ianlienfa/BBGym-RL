#ifndef NETDDPR_H
#define NETDDPR_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
#include "search_modules/Net/DDPR/NetDDPRActor.h"
#include "search_modules/Net/DDPR/NetDDPRQNet.h"
#include "user_def/oneRjSumCjNode.h"
using namespace torch;
using std::tuple;

struct StateInput
{
    const OneRjSumCjNode &node_parent;   
    const OneRjSumCjNode &node;   
    StateInput(const OneRjSumCjNode &node_parent, const OneRjSumCjNode &node) : node_parent(node_parent), node(node) {}
    vector<float> flatten_and_norm(); 
};

// the saved format should be elastic
struct ReplayBufferImpl
{
private:
    bool enter_data_prep_sec;      
public:
    vector<vector<float>> s;
    vector<float> a;
    vector<int> r;
    vector<vector<float>> s_next;
    vector<float> done;    
    int max_size;
    int idx;
    int size;

    // data section for preparation
    void enter_data_prep_section(){enter_data_prep_sec = true;};    
    void leave_data_prep_section(){enter_data_prep_sec = false;};    
    bool safe_to_submit(){return !enter_data_prep_sec;};        
    bool isin_prep(){return enter_data_prep_sec;};        
    vector<float> s_prep;
    vector<float> s_next_prep;
    float label_prep;
    float reward_prep;
    float done_prep;

    ReplayBufferImpl(int max_size);    
    const int get_size(){return size;}
    void push(vector<float> s, float a, int r, vector<float> s_, bool done);
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get(vector<int> indecies);
    void submit();
};
typedef std::shared_ptr<ReplayBufferImpl> ReplayBuffer;

struct NetDDPRImpl: nn::Cloneable<NetDDPRImpl>
{
    int64_t state_dim;
    int64_t action_dim;
    Pdd action_range;

    NetDDPRActor pi{nullptr};
    NetDDPRQNet q{nullptr};

    NetDDPRImpl(int64_t state_dim, int64_t action_dim, Pdd action_range);
    float act(torch::Tensor s);
    void reset() override
    {
        pi = register_module("PolicyNet", NetDDPRActor(state_dim, action_range));
        q = register_module("QNet", NetDDPRQNet(state_dim, action_dim));    
    }
};
TORCH_MODULE(NetDDPR);

#endif