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
    PLACE_INSERT,
    LEFT,
    RIGHT
};

struct PushBatch{
    STATE_ENCODING s;
    ACTION_ENCODING a;
    float r;
    float val;
    float logp;
};

struct SampleBatch{
    vector<float> v_s;
    vector<float> v_a;
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
    void print(){
        cout << "s_tensor: " << s << endl;
        cout << "a_tensor: " << a << endl;
        cout << "r_tensor: " << r << endl;
        cout << "adv_tensor: " << adv << endl;
        cout << "logp_tensor: " << logp << endl;
    }
};

/* extra info */
struct ExtraInfo {
    float approx_kl;
    float entropy;
    float clipfrac;
};

struct StepOutput{
    ACTION_ENCODING encoded_a;
    int64_t a;
    float v;
    float logp;
};

struct StateInput
{
    constexpr static const float norm_factor = 1e3;
    constexpr static const float zero_epsilon = 1e-11;

    const OneRjSumCjNode &node_parent;   
    const OneRjSumCjNode &node;   
    const OneRjSumCjGraph &graph;
    int state_dim = 0;
    StateInput(const OneRjSumCjNode &node_parent, const OneRjSumCjNode &node, const OneRjSumCjGraph &graph) : node_parent(node_parent), node(node), graph(graph) {}
    vector<float> flatten_and_norm(const OneRjSumCjNode &node);
    vector<float> get_state_encoding(int max_num_contour, bool get_terminal = false);
    static vector<float> get_state_encoding_fast(vector<float> &state_encoding, ::OneRjSumCjGraph &graph);
};

/*
The Replay Buffer for PPO does not mix up different epochs    
*/
struct ReplayBufferImpl
{
public:
    const float neg_epsilon = -1e-11;
    vector<STATE_ENCODING> s;
    vector<ACTION_ENCODING> a;
    vector<float> r;    
    vector<float> adv;
    vector<float> val;
    vector<float> ret;
    vector<float> logp;    
    BBARG(ReplayBufferImpl, int64_t, step, 0);          

    // prep
    struct PrepArea{
        BB_TRACK_ARG(STATE_ENCODING, s, STATE_ENCODING());
        BB_TRACK_ARG(ACTION_ENCODING, a, ACTION_ENCODING());
        BB_TRACK_ARG(float, r, numeric_limits<float>::min());
        BB_TRACK_ARG(float, val, numeric_limits<float>::min());
        BB_TRACK_ARG(float, logp, numeric_limits<float>::min());
        bool safe(){
            return s_set && a_set && r_set && val_set && logp_set;
        }
        bool empty(){
            return (!s_set) && (!a_set) && (!r_set) && (!val_set) && (!logp_set);
        }        
    } prep;

    // tracking variables
    int max_size;
    int s_feature_size;
    int a_feature_size;
    int start_idx;
    int idx;
    bool epoch_done = false;
    bool is_full = false;

    // instance-wise tracking variables
    float real_rewards = 0; 
    
    // hyperparam
    float gamma;
    float lambda;

    // data section for preparation
    bool safe_to_submit();
    void submit(bool dry_submit = false); /* for inference, use dry_submit = true to decrease memory use */
    vector<float> & vector_norm(vector<float> &vec, int start, int end);

    ReplayBufferImpl(int max_size, int batch_size);    
    int get_traj_size(){if(idx - start_idx == 0){/*cerr << "traj size = 0, might be calling at a wrong time" << endl; */} return idx - start_idx;}
    void push(const PPO::ReplayBufferImpl::PrepArea &raw_batch);
    float finish_epoch(float end_val = 0.0);
    void reset();    
    PPO::SampleBatch get();
    Batch getBatchTensor(SampleBatch &raw_batch);
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
    StepOutput step(torch::Tensor s, bool deterministic = false);
    vector<float> to_one_hot(int64_t a){
        vector<float> v;
        for(int i = 0; i < opt.action_dim; i++)
        {
            (a == i) ? v.push_back(1): v.push_back(0);
        }
        return v;
    }

    void reset() override
    {
        pi = register_module("PolicyNet", NetPPOActor(opt));    
        q = register_module("QNet", NetPPOQNet(opt));
    }
};
TORCH_MODULE(NetPPO);
};



#endif