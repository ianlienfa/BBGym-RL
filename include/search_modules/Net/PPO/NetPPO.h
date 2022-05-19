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

//                 s                a            r          s'                  //
typedef tuple<STATE_ENCODING, ACTION_ENCODING, float, STATE_ENCODING> PushBatch;
//                 s                a            r          s'        adv,    logp            //    
typedef tuple<vector<STATE_ENCODING>, vector<ACTION_ENCODING>, vector<float>, vector<STATE_ENCODING>, vector<float>, vector<float>> SampleBatch;

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

/*
The Replay Buffer for PPO does not mix up different epochs    
*/
struct ReplayBufferImpl
{
private:
    bool enter_data_prep_sec;      
public:
    vector<STATE_ENCODING> s;
    vector<ACTION_ENCODING> a;
    vector<float> r;
    vector<STATE_ENCODING> s_next;
    vector<float> adv;
    vector<float> val;
    vector<float> ret;
    vector<float> logp;

    // prep
    STATE_ENCODING s_prep;
    STATE_ENCODING s_next_prep;
    ACTION_ENCODING a_prep;
    float reward_prep;
    bool done_prep;

    // tracking variables
    int max_size;
    int s_feature_size;
    int a_feature_size;
    int start_idx;
    int idx;

    // hyperparam
    float gamma;
    float lambda;

    // data section for preparation
    void enter_data_prep_section(){enter_data_prep_sec = true;};    
    void leave_data_prep_section(){enter_data_prep_sec = false;};  
    bool isin_prep(){return enter_data_prep_sec;};          
    bool safe_to_submit();
    void submit();
    vector<float> & vector_norm(vector<float> &vec, int start, int end);

    ReplayBufferImpl(int max_size);    
    int get_size(){return idx - start_idx;}
    void push(PPO::PushBatch &raw_batch);
    void finish_epoch(float end_val = 0.0);
    PPO::SampleBatch get();            
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getBatchTensor(tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> raw_batch);
    tuple<int64_t, float, float /*a, v, logp*/> step(torch::Tensor s);
};
typedef std::shared_ptr<ReplayBufferImpl> ReplayBuffer;


struct NetPPOImpl: nn::Cloneable<NetPPOImpl>
{
    NetPPOOptions opt;
    
    NetPPOActor pi{nullptr};
    NetPPOQNet q{nullptr};
    // sample()

    NetPPOImpl(NetPPOOptions options);
    float act(torch::Tensor s);
    void reset() override
    {
        pi = register_module("PolicyNet", NetPPOActor(opt));    
        q = register_module("QNet", NetPPOQNet(opt));
    }
};
TORCH_MODULE(NetPPO);
};



#endif