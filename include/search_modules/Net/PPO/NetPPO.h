#ifndef NETPPO_H
#define NETPPO_H
#include <iostream>
#include <torch/torch.h>
#include <limits>

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

enum ACTIONS {
    PLACE,
    INSERT_PLACE,
    LEFT,
    RIGHT
};

struct PushBatch{
    STATE_ENCODING s;
    ACTION_ENCODING a;
    float r;
};

struct SampleBatch{
    vector<STATE_ENCODING> v_s;
    vector<ACTION_ENCODING> v_a;
    vector<float> v_r;
    vector<float> v_adv;
    vector<float> v_logp;
};

struct Batch{
    torch::Tensor s;
    torch::Tensor a;
    torch::Tensor r;
    torch::Tensor adv;
    torch::Tensor logp;
};

/* extra info */
struct ExtraInfo {
    float approx_kl;
    float entropy;
    float clipfrac;
};

struct StepOutput{
    int64_t a;
    float v;
    float logp;
};

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
public:
    vector<STATE_ENCODING> s;
    vector<ACTION_ENCODING> a;
    vector<float> r;    
    vector<float> adv;
    vector<float> val;
    vector<float> ret;
    vector<float> logp;

    // prep
    struct PrepArea{
        BB_TRACK_ARG(STATE_ENCODING, s, STATE_ENCODING());
        BB_TRACK_ARG(ACTION_ENCODING, a, numeric_limits<ACTION_ENCODING>::min());
        BB_TRACK_ARG(float, r, numeric_limits<float>::min());
        BB_TRACK_ARG(float, val, numeric_limits<float>::min());
        BB_TRACK_ARG(float, logp, numeric_limits<float>::min());
        // BB_TRACK_ARG(float, adv, numeric_limits<float>::min());
        // BB_TRACK_ARG(float, ret, numeric_limits<float>::min());
        bool safe(){
            return s_set && a_set && r_set && val_set && logp_set;
        }
    } prep;

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
    bool safe_to_submit();
    void submit();
    vector<float> & vector_norm(vector<float> &vec, int start, int end);

    ReplayBufferImpl(int max_size);    
    int get_size(){return idx - start_idx;}
    void push(const PPO::ReplayBufferImpl::PrepArea &raw_batch);
    void finish_epoch(float end_val = 0.0);
    void reset();
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
    StepOutput step(torch::Tensor s);
    void reset() override
    {
        pi = register_module("PolicyNet", NetPPOActor(opt));    
        q = register_module("QNet", NetPPOQNet(opt));
    }
};
TORCH_MODULE(NetPPO);
};



#endif