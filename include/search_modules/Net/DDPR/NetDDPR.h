#ifndef NETDDPR_H
#define NETDDPR_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
// #include "util/TorchUtil.h"
#include "search_modules/Net/DDPR/NetDDPRActor.h"
#include "search_modules/Net/DDPR/NetDDPRQNet.h"
#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"
using namespace torch;
using std::tuple;

struct StateInput
{
    const OneRjSumCjNode &node_parent;   
    const OneRjSumCjNode &node;   
    const OneRjSumCjGraph &graph;
    int state_dim = 0;
    StateInput(const OneRjSumCjNode &node_parent, const OneRjSumCjNode &node, const OneRjSumCjGraph &graph) : node_parent(node_parent), node(node), graph(graph) {}
    vector<float> flatten_and_norm(const OneRjSumCjNode &node);
    vector<float> get_state_encoding(bool get_terminal = false);
};

// the saved format should be elastic
struct ReplayBufferImpl
{
private:
    bool enter_data_prep_sec;      
public:
    vector<vector<float>> s;
    vector<vector<float>> a;
    vector<float> r;
    vector<vector<float>> s_next;
    vector<bool> done;    
    vector<vector<float>> contour_snapshot;
    vector<vector<float>> contour_snapshot_next;

    int max_size;
    int s_feature_size;
    int a_feature_size;
    int idx;
    int size;

    // data section for preparation
    void enter_data_prep_section(){enter_data_prep_sec = true;};    
    void leave_data_prep_section(){enter_data_prep_sec = false;};    
    bool safe_to_submit();
    bool isin_prep(){return enter_data_prep_sec;};        
    vector<float> s_prep;
    vector<float> s_next_prep;
    vector<float> a_prep;
    vector<float> contour_snapshot_prep;   
    vector<float> contour_snapshot_next_prep;    

    float reward_prep;
    bool done_prep;

    ReplayBufferImpl(int max_size);    
    int get_size(){return size;}
    void push(vector<float> s, vector<float> a, float r, vector<float> s_, bool done, vector<float> contour_snapshot, vector<float> contour_snapshot_next);
    tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> sample(vector<int> indecies);    
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getBatchTensor(tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> raw_batch);
    void submit();
};
typedef std::shared_ptr<ReplayBufferImpl> ReplayBuffer;

struct NetDDPROptions{
    int64_t state_dim;
    int64_t action_dim;
    Pdd action_range;
    string q_path = "";
    string pi_path = "";
    int64_t max_num_contour = 10000;
    int64_t rnn_hidden_size = 16;
    int64_t rnn_num_layers = 1;
};

struct NetDDPRImpl: nn::Cloneable<NetDDPRImpl>
{
    NetDDPROptions opt;
    
    NetDDPRActor pi{nullptr};
    NetDDPRQNet q{nullptr};

    NetDDPRImpl(NetDDPROptions options);
    float act(torch::Tensor s);
    void reset() override
    {
        pi = register_module("PolicyNet", NetDDPRActor(opt.state_dim, opt.action_range, opt.max_num_contour, opt.rnn_hidden_size, opt.rnn_num_layers));    
        q = register_module("QNet", NetDDPRQNet(opt.state_dim, opt.action_dim, opt.action_range, opt.max_num_contour, opt.rnn_hidden_size, opt.rnn_num_layers));
    }
};
TORCH_MODULE(NetDDPR);

#endif