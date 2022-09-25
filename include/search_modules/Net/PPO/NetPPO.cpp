#include "search_modules/Net/PPO/NetPPO.h"

PPO::ReplayBufferImpl::ReplayBufferImpl(int max_size, int batch_size) {
    this->gamma = 0.9995;
    this->lambda = 0.95;
    this->max_size = max_size;
    // this->batch_size = batch_size;
    this->idx = 0;    
    this->start_idx = 0;
    this->s_feature_size = 0;
    this->a_feature_size = 0;
    s.resize(max_size);
    a.resize(max_size);
    r.resize(max_size);
    val.resize(max_size);
    adv.resize(max_size);    
    ret.resize(max_size);
    logp.resize(max_size);
    this->reset();
}

bool PPO::ReplayBufferImpl::safe_to_submit()
{
    // do checking
    bool safe = prep.safe();
    if(safe)
    {    
        for(auto it: this->prep._s)
        {
            assertm("state variable should not be zero", it != 0);            
            assertm("state variable having too small value", it > 1e-20 || -it > 1e-20);
        }
    }
    else
    {
        #if TORCH_DEBUG >= -1
        cout << "not safe, s: " << prep._s << ", a: " << prep._a << ", r: " << prep._r << ", val: " << prep._val << ", logp: " << prep._logp << endl;
        #endif
    }
    return safe;
}        


void PPO::ReplayBufferImpl::push(const PPO::ReplayBufferImpl::PrepArea &raw_batch)
{        
    if(this->idx >= max_size)
    {
        cerr << "replay buffer index out of bound" << endl;
        exit(LOGIC_ERROR);
    }        
    if(this->idx == max_size - 1)
    {        
        this->is_full = true;
        cout << "is_full: true, this->idx: " << this->idx << "max_size: " << max_size << endl;
    }
    if(!s_feature_size)
        s_feature_size = int(s.size());
    if(!a_feature_size)
        a_feature_size = int(a.size());    
    this->s[this->idx] = raw_batch.s();
    this->a[this->idx] = raw_batch.a();
    this->r[this->idx] = raw_batch.r();
    this->val[this->idx] = raw_batch.val();
    this->logp[this->idx] = raw_batch.logp();
    this->idx = this->idx + 1;
    // debug print
    #if TORCH_DEBUG >= -1
        cout << "pushed to replay buffer, s: " << raw_batch.s() << ", a: " << raw_batch.a() << ", r: " << raw_batch.r() << ", val: " << raw_batch.val() << ", logp: " << raw_batch.logp() << endl;
    #endif
}


// To debug this: check the boundary of the value[idx] rew[idx] and adv[idx] 
// and see if the value is reasonable
// remember to fill the accu reward when calling this function
float PPO::ReplayBufferImpl::finish_epoch(float end_val)
{
    cout << "===finish epoch===" << endl;

    // dealing with full buffer
    int idx;
    if(this->is_full && idx == 0)
    {
        // this->idx = 0 in reality, but we need to use the last idx
        idx = this->max_size;
        cout << "finish epoch, idx is set to max_size, this->idx: " << this->idx << endl;
    }
    else
    {
        idx = this->idx;
    }

    cout << "start idx: " << start_idx << ", current idx: " << idx << endl;
    if(!((safe_to_submit()) || (this->prep.empty())))
    {
        assertm("calling finish_epoch when it is not safe to submit", false);
    }
    // first compute the value with the endval
    for(int i = start_idx; i < idx; i++)
    {
        adv[i] = 0.0;
    }

    // cout << "bf finish_epoch, r: " << r << endl;

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
        cout << "r[" << i << "]: " << r[i] << "+" << this->gamma << " * " << r[i+1];
        r[i] = r[i] + this->gamma * r[i + 1];
        cout << " = " << r[i] << endl;
    }

    // return accumulated discounted reward
    // cout << "after finish_epoch, r: " << r << endl;
    // turn on epoch done
    epoch_done = true;

    float return_val = r[start_idx];
    this->step() = 0;
    this->start_idx = idx;       
    
    return return_val;
}

// Debug: print mean and std
vector<float>& PPO::ReplayBufferImpl::vector_norm(vector<float> &vec, int start_idx, int idx)
{
    // dealing with full buffer    
    if(this->is_full && idx == 0)
    {
        // this->idx = 0 in reality, but we need to use the last idx
        idx = this->max_size;
        cout << "vector_norm: idx is set to max_size, this->idx: " << this->idx << endl;
    }
    else if((!this->is_full) && (idx == 1))
    {
        cout << "there's only one element in the buffer, no need to normalize" << endl;
        return vec;
    }
    else{}

    
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
    if(adv_std < 1e-30)
    {
        // cout << "adv_std is too small, set to 1e-6" << endl;
        // adv_std = 1e-6;
        cerr << "adv_std is too small, please increase the diversity of samples, try increase the epochs_per_update, early leaving" << endl;
        exit(1);
    }
    // assertm("adv_std is 0, please increase the diversity of samples, try increase the epochs_per_update", adv_std != 0);

    // normalize the advantage
    for(int i = start_idx; i < idx; i++)
    {
        adv[i] = (adv[i] - adv_mean) / adv_std;
    }    
    #if TORCH_DEBUG >= -1
    cout << "adv after norm: " << adv << endl;
    #endif
    return adv;
}

// return by constructing new array
PPO::SampleBatch PPO::ReplayBufferImpl::get()
{
    // dealing with full buffer
    int idx;
    if(this->is_full)
    {
        // this->idx = 0 in reality, but we need to use the last idx
        idx = this->max_size;
    }
    else
    {
        idx = this->idx;
    }
    
    if(!epoch_done)
        assertm("calling get when epoch is not done", false);
    else
        epoch_done = false;
    
    typedef vector<float> Vf;
    this->adv = vector_norm(this->adv, 0, idx);    
    vector<STATE_ENCODING> s = {this->s.begin(), this->s.begin() + idx};
    vector<ACTION_ENCODING> a = {this->a.begin(), this->a.begin() + idx};

    // flatten s and a
    int trajs_size = idx;
    assertm("trajs_size should be greater than 0", trajs_size > 0 && s.size() == a.size() && idx == s.size());
    vector<float> s_flat;
    vector<float> a_flat;
    for(auto it: s)
    {
        s_flat.insert(s_flat.end(), make_move_iterator(it.begin()), make_move_iterator(it.end()));
    }
    for(auto it: a)
    {
        a_flat.insert(a_flat.end(), make_move_iterator(it.begin()), make_move_iterator(it.end()));
    }

    Vf r = {this->r.begin(), this->r.begin() + idx};
    Vf adv = {this->adv.begin(), this->adv.begin() + idx};
    Vf logp = {this->logp.begin(), this->logp.begin() + idx};

    // #if TORCH_DEBUG >= -1
    cout << "s size: " << s.size() << " s: " << s << endl; 
    cout << "a size: " << a.size() << " a:  " << a << endl; 
    cout << " r size: " << r.size() << " r: " << r << endl;
    cout << " adv size: " << adv.size() << " adv: " << adv << endl;
    cout << " logp size: " << logp.size() << " logp: " << logp  << endl;    
    // #endif

    // reset
    this->idx = 0;
    this->start_idx = 0;
    this->is_full = false;
    cout << "is_full: false" << endl;

    return PPO::SampleBatch({
        .v_s = s_flat,
        .v_a = a_flat,
        .v_r = r,
        .v_adv = adv,
        .v_logp = logp
    });
}

void PPO::ReplayBufferImpl::reset()
{
    prep = PrepArea();    
}

void PPO::ReplayBufferImpl::submit(bool dry_submit)
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
        if(!dry_submit)
        {
            this->push(this->prep);             
        }
        const float reward = this->prep._r;
        this->real_rewards += reward;
        cout << "real rewards added " << reward <<  " to " << this->real_rewards << endl;
    }
    else
    {
        assertm("not safe to submit", false);
    }

    // reset temp prep vars
    this->reset();
}


vector<float> PPO::StateInput::get_state_encoding(int max_num_contour, bool get_terminal)
{    
    cout << "getting state encoding" << endl;
    vector<float> state_encoding;

    // for terminal state
    if(get_terminal)
    {
        assertm("entered get_terminal, why?", 1);
        state_encoding.assign(state_dim, 0.0);
        return state_encoding;
    }
    
    // node state
    vector<float> current_node_state = flatten_and_norm(this->node);
    cout << "current_node_state: " << current_node_state << endl;
    state_encoding.insert(state_encoding.end(), make_move_iterator(current_node_state.begin()), make_move_iterator(current_node_state.end()));

    // contour state
    vector<float> contour_snap = this->graph.get_contour_snapshot();
    cout << "contour_snap: " << contour_snap << endl;
    state_encoding.insert(state_encoding.end(), make_move_iterator(contour_snap.begin()), make_move_iterator(contour_snap.end()));

    // current_pointer
    cout << "current_pointer: " << this->graph.contours.current_pos << endl;
    const float current_contour_pointer = this->graph.contours.current_pos / norm_factor + zero_epsilon;
    state_encoding.push_back(current_contour_pointer);

    // picker steps
    cout << "picker_steps: " << this->graph.contours.picker_steps << endl;
    const float picker_steps = (float(this->graph.contours.picker_steps)) / norm_factor + zero_epsilon;
    state_encoding.push_back(picker_steps);

    // picker_pointer
    cout << "picker_pointer: " << this->graph.contours.picker_pos << endl;
    const float current_picker_pos = this->graph.contours.picker_pos / norm_factor + zero_epsilon;
    state_encoding.push_back(current_picker_pos);

    // initialize the state encoding dimension
    if(state_dim == 0) state_dim = state_encoding.size();    
    return state_encoding;
}

vector<float> PPO::StateInput::get_state_encoding_fast(vector<float> &state_encoding, ::OneRjSumCjGraph &graph)
{
    auto idx_end = state_encoding.size() - 1;

    // picker steps
    const float picker_steps = (float(graph.contours.picker_steps)) / norm_factor + zero_epsilon;
    state_encoding[idx_end - 1] = picker_steps;

    // picker pointer
    vector<float> state_encoding_copy = state_encoding;
    auto current_picker_pos = graph.contours.picker_pos / norm_factor + zero_epsilon;
    state_encoding[idx_end] = current_picker_pos;
    cout << "fast state encoding: " << state_encoding << endl;
    return state_encoding;
}

vector<float> PPO::StateInput::flatten_and_norm(const OneRjSumCjNode &node)
{   
    float state_processed_rate = 0.0,
        norm_lb = 0.0,
        norm_weighted_completion_time = 0.0,
        norm_current_feasible_solution = 0.0
    ;
    const float epsilon = 1e-6;
    vector<float> node_state_encoding;

    cout << "==== debugging flatten and norm ===" << endl;
    
    // state_processed_rate
    state_processed_rate = ((float)(node.seq.size())+epsilon) / (float) OneRjSumCjNode::jobs_num;
    node_state_encoding.push_back(state_processed_rate);

    cout << "node seq size: " << node.seq.size() << endl;

    // norm_lb    
    norm_lb = (node.lb + epsilon) / (float) OneRjSumCjNode::worst_upperbound;    
    node_state_encoding.push_back(norm_lb);

    cout << "norm_lb: " << node.lb << ", epsilon: " << epsilon << ", worst_upperbound: " << OneRjSumCjNode::worst_upperbound << endl;

    // norm_weighted_completion_timeπ
    norm_weighted_completion_time = (node.weighted_completion_time+epsilon)  / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_weighted_completion_time);

    cout << "norm_weighted_completion_time: " << node.weighted_completion_time << ", epsilon: " << epsilon << ", worst_upperbound: " << OneRjSumCjNode::worst_upperbound << endl;

    // norm_feasible_solution
    norm_current_feasible_solution = (graph.min_obj+epsilon)  / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_current_feasible_solution);

    cout << "norm_current_feasible_solution: " << graph.min_obj << ", epsilon: " << epsilon << ", worst_upperbound: " << OneRjSumCjNode::worst_upperbound << endl;

    cout << "==== end debugging flatten and norm ===" << endl;

    return node_state_encoding;
}

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

PPO::Batch PPO::ReplayBufferImpl::getBatchTensor(PPO::SampleBatch &raw_batch)
{
    using std::get, std::make_tuple;
        
    vector<float>& s_flat = raw_batch.v_s;
    vector<float>& a_flat = raw_batch.v_a;
    vector<float>& r_flat = raw_batch.v_r;
    vector<float>& adv_flat = raw_batch.v_adv;
    vector<float>& logp_flat = raw_batch.v_logp;    
    
    // do checking
    
    for(auto it: s_flat)
    {        
        if(it == 0 || !(it > 1e-20 || it < -1e-20))
            cout << "state variable " << s_flat << "it: " << it << endl;
        assertm("state variable should not be zero", it != 0);                            
        assertm("state variable having too small value", (it > 1e-20 || it < -1e-20));
    }

    int64_t state_feature_size = (int64_t)(this->s[0].size());
    int64_t action_feature_size = (int64_t)(this->a[0].size());    // one int as its action feature

    int64_t batch_size = r_flat.size();   
    
    // assertm("batch size should be identical", batch_size == (s_flat_sz / s_feature_size));
    // assertm("batch size should be identical", batch_size == (a_flat_sz / a_feature_size));
    assertm("batch size should be identical", (r_flat.size() == adv_flat.size() && r_flat.size() == logp_flat.size()));
    // turn arrays to Tensor
    Tensor s_tensor = torch::from_blob(s_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor a_tensor = torch::from_blob(a_flat.data(), {batch_size, action_feature_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor r_tensor = torch::from_blob(r_flat.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor adv_tensor = torch::from_blob(adv_flat.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor logp_tensor = torch::from_blob(logp_flat.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    return PPO::Batch({
        .s = s_tensor,
        .a = a_tensor,
        .r = r_tensor,
        .adv = adv_tensor,
        .logp = logp_tensor
    });
}


PPO::NetPPOImpl::NetPPOImpl(NetPPOOptions opt)
{
    this->opt = opt;        
    assertm("state_dim should be greater than 0", opt.state_dim > 0);
    assertm("action_dim should be greater than 0", opt.action_dim > 0);    
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

PPO::StepOutput PPO::NetPPOImpl::step(torch::Tensor s, bool deterministic)
{
    GRAD_TOGGLE(this->pi, false);
    GRAD_TOGGLE(this->q, false);
    
    // sample from the softmax tensor
    torch::Tensor dist_out = this->pi->dist(s);
    if(dist_out.isnan().any().item<bool>())
    {
        cerr << "s: " << s << endl;
        cerr << "dist_out output dist in step(): " << dist_out << endl;
        assertm("dist_out output dist is nan", false);
    }
    int64_t a;    
    if(deterministic)
    {
        a = torch::argmax(dist_out).item<int64_t>();
    }
    else
    {
        a = torch::multinomial(dist_out, 1).item().toLong();
    }

    // if(deterministic){
        // cout << "s: " << s << endl;
        cout << "dist_out: " << dist_out << endl;
        cout << "a: " << a << endl;
    // }
    auto encoded_a = to_one_hot(a);
    assertm("action should be in range", ((a >= 0) && (a < opt.action_dim)));
    float logp = torch::log(dist_out[0][a]).item().toFloat();
    float v = this->q->forward(s).item().toFloat();

    GRAD_TOGGLE(this->pi, true);
    GRAD_TOGGLE(this->q, true);

    return StepOutput({
        .encoded_a = encoded_a,
        .a = a,        
        .v = v,
        .logp = logp    
    });
}


