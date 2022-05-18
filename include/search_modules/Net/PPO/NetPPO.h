#ifndef NETPPO_H
#define NETPPO_H
#include <iostream>
#include <torch/torch.h>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
// #include "util/TorchUtil.h"
#include "search_modules/Net/PPO/PPO.h"
#include "search_modules/Net/PPO/NetPPOActor.h"
#include "search_modules/Net/PPO/NetPPOQNet.h"
#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"
using namespace torch;
using std::tuple;

namespace PPO {
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
};

struct NetPPOImpl: nn::Cloneable<NetPPOImpl>
{
    NetPPOOptions opt;
    
    NetPPOActor pi{nullptr};
    NetPPOQNet q{nullptr};

    NetPPOImpl(NetPPOOptions options);
    float act(torch::Tensor s);
    void reset() override
    {
        pi = register_module("PolicyNet", NetPPOActor(opt));    
        q = register_module("QNet", NetPPOQNet(opt));
    }
};
TORCH_MODULE(NetPPO);

#endif