#include "search_modules/Net/PPO/NetPPO.h"

PPO::ReplayBufferImpl::ReplayBufferImpl(int max_size) {

    this->gamma = 0.99;
    this->lambda = 0.95;

    this->max_size = max_size;
    this->idx = 0;    
    this->start_idx = 0;
    this->s_feature_size = 0;
    this->a_feature_size = 0;
    this->enter_data_prep_sec = false;
    s.resize(max_size);
    a.resize(max_size);
    r.resize(max_size);
    s_next.resize(max_size);
    adv.resize(max_size);    
    ret.resize(max_size);
    logp.resize(max_size);
}

bool PPO::ReplayBufferImpl::safe_to_submit()
{
    // do checking
    for(auto it: this->s_prep)
    {
        assertm("state variable should not be zero", it != 0);
        assertm("state variable having too small value", it > 1e-20);
    }
    for(auto it: this->s_next_prep)
    {
        assertm("state variable should not be zero", it != 0);
        assertm("state variable having too small value", it > 1e-20);
    }
    assertm("done variable should be 0 or 1", (this->done_prep == 0.0 || this->done_prep == 1.0));
    assertm("state_vector should not be empty", (!this->s_prep.empty()));
    assertm("state_next_vector should not be empty", (!this->s_next_prep.empty()));
    assertm("action_vector should not be empty", (!this->a_prep.empty()));        
    return !enter_data_prep_sec;   
}        


//                 s                a            r          s'                  //
//typedef tuple<STATE_ENCODING, ACTION_ENCODING, float, STATE_ENCODING> PushBatch;//
void PPO::ReplayBufferImpl::push(PPO::PushBatch &raw_batch)
{    
    if(idx >= max_size)
    {
        cout << "replay buffer index out of bound" << endl;
        exit(LOGIC_ERROR);
    }    
    if(!s_feature_size)
        s_feature_size = int(s.size());
    if(!a_feature_size)
        a_feature_size = int(a.size());
    // s and s_next have no strict relation on the sequence 
    this->s[this->idx] = std::get<0>(raw_batch);
    this->a[this->idx] = std::get<1>(raw_batch);
    this->r[this->idx] = std::get<2>(raw_batch);
    this->s_next[this->idx] = std::get<3>(raw_batch);
    this->idx = (this->idx + 1) % max_size;    
}


// To debug this: check the boundary of the value[idx] rew[idx] and adv[idx] 
// and see if the value is reasonable
void PPO::ReplayBufferImpl::finish_epoch(float end_val)
{
    if(!safe_to_submit())
    {
        cout << "calling finish_epoch when it is not safe to submit" << endl;
        assert(false);
    }

    // first compute the value with the endval
    for(int i = start_idx; i < idx; i++)
    {
        adv[i] = 0.0;
    }
    adv[idx - 1] = r[idx - 1] + this->gamma * end_val - val[idx - 1];
    r[idx - 1] = gamma * end_val + r[idx - 1];

    // then compute the other value without endval
    for(int i = idx - 2; i >= start_idx; i--)
    {   
        float delta = r[i] + this->gamma * val[i+1] - val[i];
        adv[i] = delta + this->gamma * this->lambda * adv[i+1];
    }   

    // compute accumulated discounted reward for MSE
    for(int i = idx - 2; i >= start_idx; i--)
    {
        r[i] = r[i] + this->gamma * r[i + 1];
    }
}

// Debug: print mean and std
vector<float>& PPO::ReplayBufferImpl::vector_norm(vector<float> &vec, int start_idx, int idx)
{
    vector<float> &adv = vec;
    float adv_mean = 0.0, adv_sum = 0.0, adv_std = 0.0;
    for(int i = start_idx; i < idx; i++)
    {
        adv_sum += adv[i];
    }
    adv_mean = adv_sum / (idx - start_idx);
    for(int i = start_idx; i < idx; i++)
    {
        adv_std += (adv[i] - adv_mean) * (adv[i] - adv_mean);
    }
    adv_std /= (idx - start_idx);
    adv_std = sqrt(adv_std);

    // normalize the advantage
    for(int i = start_idx; i < idx; i++)
    {
        adv[i] = (adv[i] - adv_mean) / adv_std;
    }    
    return adv;
}

// return by constructing new array
PPO::SampleBatch PPO::ReplayBufferImpl::get()
{
    typedef vector<float> Vf;
    adv = vector_norm(adv, start_idx, idx);
    vector<STATE_ENCODING> s = {this->s.begin() + start_idx, this->s.begin() + idx};
    vector<ACTION_ENCODING> a = {this->a.begin() + start_idx, this->a.begin() + idx};
    Vf r = {this->r.begin() + start_idx, this->r.begin() + idx};
    vector<STATE_ENCODING> s_next = {this->s_next.begin() + start_idx, this->s_next.begin() + idx};
    Vf adv = {this->adv.begin() + start_idx, this->adv.begin() + idx};
    Vf logp = {this->logp.begin() + start_idx, this->logp.begin() + idx};
    return PPO::SampleBatch(s, a, r, s_next, adv, logp);
}

void PPO::ReplayBufferImpl::submit()
{    
    #if TORCH_DEBUG == 1    
        cout << "label buffer dynamics: " << endl; 
        cout << "idx: " << idx << endl;                   
        for(int i = 0; i < 50; i++)
        {
            cout << i << ": " << a[i] << endl;
        }
    #endif

    if(safe_to_submit())
    {        
        cout << "a_prep: " << a_prep << endl;
        this->push(PPO::PushBatch(this->s_prep, this->a_prep, this->reward_prep, this->s_next_prep));
    }
    else
    {
        cout << "not safe to submit" << endl;
        exit(LOGIC_ERROR);
    }
}


// vector<float> PPO::StateInput::get_state_encoding(bool get_terminal)
// {    
//     vector<float> state_encoding;

//     // for terminal state
//     if(get_terminal)
//     {
//         assertm("entered get_terminal, why?", 1);
//         state_encoding.assign(state_dim, 0.0);
//         return state_encoding;
//     }

//     vector<float> current_node_state = flatten_and_norm(this->node);
//     // vector<float> parent_node_state = flatten_and_norm(this->node_parent);
//     state_encoding.insert(state_encoding.end(), make_move_iterator(current_node_state.begin()), make_move_iterator(current_node_state.end()));
//     // state_encoding.insert(state_encoding.end(), make_move_iterator(parent_node_state.begin()), make_move_iterator(parent_node_state.end()));

//     // initialize the state encoding dimension
//     if(state_dim == 0) state_dim = state_encoding.size();    
//     return state_encoding;
// }

// vector<float> PPO::StateInput::flatten_and_norm(const OneRjSumCjNode &node)
// {   
//     float state_processed_rate = 0.0,
//         norm_lb = 0.0,
//         norm_weighted_completion_time = 0.0,
//         norm_current_feasible_solution = 0.0
//     ;
//     const float epsilon = 1e-6;
//     vector<float> node_state_encoding;
    
//     // state_processed_rate
//     state_processed_rate = ((float)(node.seq.size())+epsilon) / (float) OneRjSumCjNode::jobs_num;
//     node_state_encoding.push_back(state_processed_rate);

//     // norm_lb    
//     norm_lb = (node.lb + epsilon) / (float) OneRjSumCjNode::worst_upperbound;    
//     node_state_encoding.push_back(norm_lb);

//     // norm_weighted_completion_timeπ
//     norm_weighted_completion_time = (node.weighted_completion_time+epsilon)  / (float) OneRjSumCjNode::worst_upperbound;
//     node_state_encoding.push_back(norm_weighted_completion_time);

//     // norm_feasible_solution
//     norm_current_feasible_solution = (graph.min_obj+epsilon)  / (float) OneRjSumCjNode::worst_upperbound;
//     node_state_encoding.push_back(norm_current_feasible_solution);

//     return node_state_encoding;
// }

// tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> PPO::ReplayBufferImpl::sample(vector<int> indecies)
// {
//     int batch_size = indecies.size();
//     vector<vector<float>> s;
//     vector<vector<float>> a;
//     vector<float> r;
//     vector<vector<float>> s_next;
//     vector<bool> done;
//     vector<vector<float>> contour_snapshot;
//     vector<vector<float>> contour_snapshot_next;

//     for(int i = 0; i < indecies.size(); i++)
//     {
//         int idx = indecies[i];
//         if(idx >= this->size)
//             throw std::out_of_range("buffer access index out of bound");
//         s.push_back(this->s[idx]);
//         a.push_back(this->a[idx]);
//         r.push_back(this->r[idx]);
//         s_next.push_back(this->s_next[idx]);
//         done.push_back(this->done[idx]);
//         contour_snapshot.push_back(this->contour_snapshot[idx]);
//         contour_snapshot_next.push_back(this->contour_snapshot_next[idx]);
//     }

//     // sampled action
//     cout << "sampled action: " << endl;
//     for(auto it: a)
//     {
//         cout << it << endl;
//     }

//     assertm("batch size should be identical", (s.size() == a.size() && s.size() == r.size() && s.size() == s_next.size() && s.size() == done.size() && s.size() == batch_size));

//     // flatten the state array
//     vector<float> s_flat;
//     vector<float> s_next_flat;
//     for(int i = 0; i < batch_size; i++)
//     {
//         s_flat.insert(s_flat.end(), make_move_iterator(s[i].begin()), make_move_iterator(s[i].end()));
//         s_next_flat.insert(s_next_flat.end(), make_move_iterator(s_next[i].begin()), make_move_iterator(s_next[i].end()));
//     }

//     int state_dim = this->s[0].size();
//     assertm("state size should be identical", (s_flat.size() == s_next_flat.size() && s_flat.size() == state_dim * batch_size));
    

//     vector<float> action_flat;
//     for(int i = 0; i < batch_size; i++)
//     {
//         action_flat.insert(action_flat.end(), make_move_iterator(a[i].begin()), make_move_iterator(a[i].end()));
//     }
//     #ifndef NDEBUG
//     for(int i = 0; i < batch_size - 1; i++)
//     {
//         assertm("contour size should be identical", contour_snapshot[i].size() == contour_snapshot[i+1].size());
//     }
//     for(int i = 0; i < batch_size - 1; i++)
//     {
//         assertm("contour size should be identical", contour_snapshot_next[i].size() == contour_snapshot_next[i+1].size());
//     }
//     #endif
//     vector<float> contour_snapflat;
//     vector<float> contour_snapflat_next; 
//     for(int i = 0; i < batch_size; i++)
//     {
//         contour_snapflat.insert(contour_snapflat.end(), make_move_iterator(contour_snapshot[i].begin()), make_move_iterator(contour_snapshot[i].end()));
//     }
//     for(int i = 0; i < batch_size; i++)
//     {
//         contour_snapflat_next.insert(contour_snapflat_next.end(), make_move_iterator(contour_snapshot_next[i].begin()), make_move_iterator(contour_snapshot_next[i].end()));
//     }
//     assertm("In batch, contour and next contour size should be identical", contour_snapflat.size() == contour_snapflat_next.size());
//     return make_tuple(s_flat, action_flat, r, s_next_flat, done, contour_snapflat, contour_snapflat_next);
// }

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PPO::ReplayBufferImpl::getBatchTensor(tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> raw_batch)
{
    using std::get, std::make_tuple;
    
    vector<float> s_flat;
    vector<float> s_next_flat;
    vector<float> a;
    vector<float> r;
    vector<bool> done;
    vector<float> contour_snapflat;
    vector<float> contour_snapflat_next;


    s_flat = get<0>(raw_batch);
    a = get<1>(raw_batch);
    r = get<2>(raw_batch);
    s_next_flat = get<3>(raw_batch);
    done = get<4>(raw_batch);
    contour_snapflat = get<5>(raw_batch);
    contour_snapflat_next = get<6>(raw_batch);

    
    // do checking
    for(auto it: s_flat)
    {
        assertm("state variable should not be zero", it != 0);
        assertm("state variable having too small value", it > 1e-20);
    }
    for(auto it: s_next_flat)
    {
        assertm("state variable should not be zero", it != 0);
        assertm("state variable having too small value", it > 1e-20);
    }
    for(auto it: done)
    {
        assertm("done variable should be 0 or 1", (it == 0.0 || it == 1.0));
    }

    int batch_size = done.size();
    int state_feature_size = this->s[0].size();
    int action_feature_size = this->a[0].size();
    int contour_snapshot_feature_size = this->contour_snapshot[0].size();
    assertm("state feature size should be same", this->s[0].size() == this->s_next[0].size());
    assertm("contour feature size should be same", this->contour_snapshot[0].size() == this->contour_snapshot_next[0].size());    

    // turn arrays to Tensor
    Tensor s_tensor = torch::from_blob(s_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor a_tensor = torch::from_blob(a.data(), {batch_size, action_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor r_tensor = torch::from_blob(r.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor s_next_tensor = torch::from_blob(s_next_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    bool cArr_done[batch_size];
    for(int i = 0; i < batch_size; i++)
        cArr_done[i] = done[i];    
    Tensor done_tensor = torch::from_blob(cArr_done, {batch_size, 1}, torch::TensorOptions().dtype(torch::kBool)).clone();
    Tensor contour_snapshot_tensor = torch::from_blob(contour_snapflat.data(), {batch_size, contour_snapshot_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor contour_snapshot_next_tensor = torch::from_blob(contour_snapflat_next.data(), {batch_size, contour_snapshot_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return make_tuple(s_tensor, a_tensor, r_tensor, s_next_tensor, done_tensor, contour_snapshot_tensor, contour_snapshot_next_tensor);
}

tuple<int64_t, float, float /*a, v, logp*/> NetPPOImpl::step(torch::Tensor s)
{    
    torch::NoGradGuard no_grad;
    int64_t a = 0;
    float v = 0, logp = 0;
    torch::Tensor pi = this->pi->dist(s);
    a = torch::multinomial(pi, 1).item().toLong();
    v = this->q->forward(s).item().toFloat();
}

NetPPOImpl::NetPPOImpl(NetPPOOptions opt)
{
    this->opt = opt;        
    assertm("state_dim should be greater than 0", opt.state_dim > 0);
    assertm("action_dim should be greater than 0", opt.action_dim > 0);
    assertm("max_num_contour should be greater than 0", opt.max_num_contour > 0);
    assertm("rnn_hidden_size should be greater than 0", opt.rnn_hidden_size > 0);
    assertm("rnn_num_layers should be greater than 0", opt.rnn_num_layers > 0);
    NetPPOQNet q_net(opt);
    NetPPOActor pi_net(opt);    

    if(opt.q_path != "" && opt.pi_path != "")
    {
        cout << "loading saved model from: " << opt.q_path << " and " << opt.pi_path << endl;
        torch::load(q_net, opt.q_path);
        torch::load(pi_net, opt.pi_path);

        print_modules(*q_net);
        print_modules(*pi_net);
    }
    
    this->q = register_module("QNet", q_net);
    this->pi = register_module("PolicyNet", pi_net);
}



float NetPPOImpl::act(torch::Tensor s)
{
    // torch::NoGradGuard no_grad;
    // #if DEBUG_LEVEL >= 2
    // for(const auto &p: this->q->parameters())
    // {
    //     cout << p.requires_grad() << endl;
    // }
    // #endif
    // return this->pi->forward(s)[2].item<float>();
}

