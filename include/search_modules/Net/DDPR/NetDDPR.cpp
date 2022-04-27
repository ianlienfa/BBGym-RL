#include "search_modules/Net/DDPR/NetDDPR.h"

ReplayBufferImpl::ReplayBufferImpl(int max_size) {
    this->max_size = max_size;
    this->idx = 0;
    this->size = 0;
    this->s_feature_size = 0;
    this->a_feature_size = 0;
    this->enter_data_prep_sec = false;
    s.resize(max_size);
    a.resize(max_size);
    r.resize(max_size);
    s_next.resize(max_size);
    done.resize(max_size);
    contour_snapshot.resize(max_size);
    contour_snapshot_next.resize(max_size);
}

bool ReplayBufferImpl::safe_to_submit()
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
    assertm("contour_snapshot_vector should not be empty", (!this->contour_snapshot_prep.empty()));
    assertm("contour_snapshot_next_vector should not be empty", (!this->contour_snapshot_next_prep.empty()));
    return !enter_data_prep_sec;   
}        

void ReplayBufferImpl::push(vector<float> s, vector<float> a, float r, vector<float> s_, bool done, vector<float> contour_snapshot, vector<float> contour_snapshot_next)
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
    this->s[this->idx] = s;
    this->a[this->idx] = a;
    this->r[this->idx] = r;
    this->s_next[this->idx] = s_;
    this->done[this->idx] = done;
    this->contour_snapshot[this->idx] = contour_snapshot;
    this->contour_snapshot_next[this->idx] = contour_snapshot_next;

    this->idx = (this->idx + 1) % max_size;
    this->size = std::min(this->size + 1, max_size);
}

void ReplayBufferImpl::submit()
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
        this->push(this->s_prep, this->a_prep, this->reward_prep, this->s_next_prep, this->done_prep, this->contour_snapshot_prep, this->contour_snapshot_next_prep);
    }
    else
    {
        cout << "not safe to submit" << endl;
        exit(LOGIC_ERROR);
    }
}



vector<float> StateInput::get_state_encoding(bool get_terminal)
{    
    vector<float> state_encoding;

    // for terminal state
    if(get_terminal)
    {
        state_encoding.assign(state_dim, 0.0);
        return state_encoding;
    }

    vector<float> current_node_state = flatten_and_norm(this->node);
    vector<float> parent_node_state = flatten_and_norm(this->node_parent);
    state_encoding.insert(state_encoding.end(), make_move_iterator(current_node_state.begin()), make_move_iterator(current_node_state.end()));
    state_encoding.insert(state_encoding.end(), make_move_iterator(parent_node_state.begin()), make_move_iterator(parent_node_state.end()));

    // initialize the state encoding dimension
    if(state_dim == 0) state_dim = state_encoding.size();    
    return state_encoding;
}

vector<float> StateInput::flatten_and_norm(const OneRjSumCjNode &node)
{   
    float state_processed_rate = 0.0,
        norm_lb = 0.0,
        norm_weighted_completion_time = 0.0,
        norm_current_feasible_solution = 0.0
    ;

    vector<float> node_state_encoding;

    // state_processed_rate
    state_processed_rate = ((float)(node.seq.size())+1e-6) / (float) OneRjSumCjNode::jobs_num;
    node_state_encoding.push_back(state_processed_rate);

    // norm_lb    
    norm_lb = (node.lb + 1e-6) / (float) OneRjSumCjNode::worst_upperbound;    
    node_state_encoding.push_back(norm_lb);

    // norm_weighted_completion_timeπ
    norm_weighted_completion_time = (node.weighted_completion_time+1e-6)  / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_weighted_completion_time);

    // norm_feasible_solution
    norm_current_feasible_solution = (graph.min_obj+1e-6)  / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_current_feasible_solution);

    return node_state_encoding;
}

tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> ReplayBufferImpl::sample(vector<int> indecies)
{
    int batch_size = indecies.size();
    vector<vector<float>> s;
    vector<vector<float>> a;
    vector<float> r;
    vector<vector<float>> s_next;
    vector<bool> done;
    vector<vector<float>> contour_snapshot;
    vector<vector<float>> contour_snapshot_next;

    cout << "contour_snapshot: " << contour_snapshot.size() << endl;
    for(auto it: contour_snapshot){
        cout << it << endl;
    }
    cout << "contour_snapshot_next: " << contour_snapshot_next.size() << endl;
    for(auto it: contour_snapshot_next){
        cout << it << endl;
    }    

    for(int i = 0; i < indecies.size(); i++)
    {
        int idx = indecies[i];
        if(idx >= this->size)
            throw std::out_of_range("buffer access index out of bound");
        s.push_back(this->s[idx]);
        a.push_back(this->a[idx]);
        r.push_back(this->r[idx]);
        s_next.push_back(this->s_next[idx]);
        done.push_back(this->done[idx]);
        contour_snapshot.push_back(this->contour_snapshot[idx]);
        contour_snapshot_next.push_back(this->contour_snapshot_next[idx]);
    }

    assertm("batch size should be identical", (s.size() == a.size() && s.size() == r.size() && s.size() == s_next.size() && s.size() == done.size() && s.size() == batch_size));

    // flatten the state array
    vector<float> s_flat;
    vector<float> s_next_flat;
    for(int i = 0; i < batch_size; i++)
    {
        s_flat.insert(s_flat.end(), make_move_iterator(s[i].begin()), make_move_iterator(s[i].end()));
        s_next_flat.insert(s_next_flat.end(), make_move_iterator(s_next[i].begin()), make_move_iterator(s_next[i].end()));
    }

    int state_dim = this->s[0].size();
    assertm("state size should be identical", (s_flat.size() == s_next_flat.size() && s_flat.size() == state_dim * batch_size));

    vector<float> action_flat;
    for(int i = 0; i < batch_size; i++)
    {
        action_flat.insert(action_flat.end(), make_move_iterator(a[i].begin()), make_move_iterator(a[i].end()));
    }
    #ifndef NDEBUG
    for(int i = 0; i < batch_size - 1; i++)
    {
        assertm("contour size should be identical", contour_snapshot[i].size() == contour_snapshot[i+1].size());
    }
    for(int i = 0; i < batch_size - 1; i++)
    {
        assertm("contour size should be identical", contour_snapshot_next[i].size() == contour_snapshot_next[i+1].size());
    }
    #endif
    vector<float> contour_snapflat;
    vector<float> contour_snapflat_next; 
    for(int i = 0; i < batch_size; i++)
    {
        contour_snapflat.insert(contour_snapflat.end(), make_move_iterator(contour_snapshot[i].begin()), make_move_iterator(contour_snapshot[i].end()));
    }
    cout << "printing contour_snapshot: " << contour_snapshot.size() << endl;    
    for(int i = 0; i < batch_size; i++)
    {
        cout << contour_snapshot[i] << endl;
        contour_snapflat_next.insert(contour_snapflat_next.end(), make_move_iterator(contour_snapshot_next[i].begin()), make_move_iterator(contour_snapshot_next[i].end()));
    }
    assertm("In batch, contour and next contour size should be identical", contour_snapflat.size() == contour_snapflat_next.size());
    return make_tuple(s_flat, action_flat, r, s_next_flat, done, contour_snapflat, contour_snapflat_next);
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ReplayBufferImpl::getBatchTensor(tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<bool>, vector<float>, vector<float>> raw_batch)
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
    Tensor contour_snapshot_tensor = torch::from_blob(contour_snapflat.data(), {batch_size, contour_snapshot_feature_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    Tensor contour_snapshot_next_tensor = torch::from_blob(contour_snapflat_next.data(), {batch_size, contour_snapshot_feature_size, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return make_tuple(s_tensor, a_tensor, r_tensor, s_next_tensor, done_tensor, contour_snapshot_tensor, contour_snapshot_next_tensor);
}


NetDDPRImpl::NetDDPRImpl(NetDDPROptions opt)
{
    this->opt = opt;        
    assertm("state_dim should be greater than 0", opt.state_dim > 0);
    assertm("action_dim should be greater than 0", opt.action_dim > 0);
    assertm("max_num_contour should be greater than 0", opt.max_num_contour > 0);
    assertm("rnn_hidden_size should be greater than 0", opt.rnn_hidden_size > 0);
    assertm("rnn_num_layers should be greater than 0", opt.rnn_num_layers > 0);
    NetDDPRQNet q_net(opt.state_dim, opt.action_dim, opt.action_range, opt.max_num_contour, opt.rnn_hidden_size, opt.rnn_num_layers);
    NetDDPRActor pi_net(opt.state_dim, opt.action_range, opt.max_num_contour, opt.rnn_hidden_size, opt.rnn_num_layers);    

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



float NetDDPRImpl::act(torch::Tensor s)
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

