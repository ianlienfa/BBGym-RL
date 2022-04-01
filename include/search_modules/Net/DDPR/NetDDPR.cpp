#include "search_modules/Net/DDPR/NetDDPR.h"

ReplayBufferImpl::ReplayBufferImpl(int max_size) {
    this->max_size = max_size;
    this->idx = 0;
    this->size = 0;
    this->s_feature_size = 0;
    this->enter_data_prep_sec = false;
    s.resize(max_size);
    a.resize(max_size);
    r.resize(max_size);
    s_next.resize(max_size);
    done.resize(max_size);
}

void ReplayBufferImpl::push(vector<float> s, float a, int r, vector<float> s_, bool done)
{    
    #if TORCH_DEBUG == 1                        
    if(std::isnan(a))
        throw std::runtime_error("Labeler returned NaN");
    #endif

    if(idx >= max_size)
    {
        cout << "replay buffer index out of bound" << endl;
        exit(LOGIC_ERROR);
    }    
    if(!s_feature_size)
        s_feature_size = int(s.size());
    // s and s_next have no strict relation on the sequence 
    this->s[this->idx] = s;
    this->a[this->idx] = a;
    this->r[this->idx] = r;
    this->s_next[this->idx] = s_;
    this->done[this->idx] = done;
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
        this->push(this->s_prep, this->label_prep, this->reward_prep, this->s_next_prep, this->done_prep);
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
    vector<float> parent_node_state = flatten_and_norm(this->node);
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
    state_processed_rate = node.seq.size() / (float) OneRjSumCjNode::jobs_num;
    node_state_encoding.push_back(state_processed_rate);

    // norm_lb
    norm_lb = node.lb / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_lb);

    // norm_weighted_completion_time
    norm_weighted_completion_time = node.weighted_completion_time / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_weighted_completion_time);

    // norm_feasible_solution
    norm_current_feasible_solution = graph.min_obj / (float) OneRjSumCjNode::worst_upperbound;
    node_state_encoding.push_back(norm_current_feasible_solution);

    return node_state_encoding;
}

// const tuple<vector<vector<float>>, vector<float>, vector<float>, vector<vector<float>>, vector<float>> & sample(vector<int> indecies)
// {
//        // cout << "printing s: " << endl;
//     // for(int i = 0; i < this->idx; i++)
//     // {
//     //     cout << "{ ";
//     //     for(auto it2: this->s[i])
//     //     {
//     //         cout << it2 << " ";
//     //     }
//     //     cout << "}" << endl;
//     // }

//     int batch_size = indecies.size();
//     int state_feature_size = this->s[0].size();
//     vector<vector<float>> s;
//     vector<float> a;
//     vector<float> r;
//     vector<vector<float>> s_next;
//     vector<float> done;
//     for(int i = 0; i < indecies.size(); i++)
//     {
//         int idx = indecies[i];
//         if(idx > this->max_size)
//             throw std::out_of_range("buffer access index out of bound");
//         s.push_back(this->s[idx]);
//         a.push_back(this->a[idx]);
//         r.push_back(this->r[idx]);
//         s_next.push_back(this->s_next[idx]);
//         done.push_back(this->done[idx]);
//     }
//     // flatten the state array
//     vector<float> s_flat;
//     vector<float> s_next_flat;
//     for(int i = 0; i < batch_size; i++)
//     {
//         s_flat.insert(s_flat.end(), make_move_iterator(s[i].begin()), make_move_iterator(s[i].end()));
//         s_next_flat.insert(s_next_flat.end(), make_move_iterator(s_next[i].begin()), make_move_iterator(s_next[i].end()));
//     }

//     cout << "The chosen action input is: " << endl;
//     for(auto it: a)
//         cout << it << " ";
//     cout << endl;

//     return make_tuple(s_flat, a, r, s_next_flat, done);
// }

// tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> getBatchTensor(const RawBatch & raw_batch)
// {
//     vector<vector<float>> s;
//     vector<float> a;
//     vector<float> r;
//     vector<vector<float>> s_next;
//     vector<float> done;
//     std::tie(s, a, r, s_next, done) = raw_batch;

// }


// should output in tensor form
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ReplayBufferImpl::get(vector<int> indecies)
{   
 
//     // turn arrays to Tensor
//     Tensor s_tensor = torch::from_blob(s_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32));
//     Tensor a_tensor = torch::from_blob(a.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32));
//     Tensor r_tensor = torch::from_blob(r.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32));
//     Tensor s_next_tensor = torch::from_blob(s_next_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32));
//     Tensor done_tensor = torch::from_blob(done.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kBool));

//     // cout << "state_feature_size: " << state_feature_size << endl;
//     // cout << "s_tensor: " << s_tensor << endl;
//     // cout << "s_next_tensor: " << s_next_tensor.sizes() << endl;
//     cout << "a_tensor: " << a_tensor << endl;
//     return std::make_tuple(s_tensor, a_tensor, r_tensor, s_next_tensor, done_tensor);
// }
}


NetDDPRImpl::NetDDPRImpl(int64_t state_dim, int64_t action_dim, Pdd action_range, string q_path, string pi_path)
{
    this->state_dim = state_dim;
    this->action_dim = action_dim;
    this->action_range = action_range;

    NetDDPRQNet q_net(state_dim, action_dim);
    // module_info(*q_net);
    NetDDPRActor pi_net(state_dim, action_range);
    // module_info(*pi_net);

    if(q_path != "" && pi_path != "")
    {
        torch::load(q_net, q_path);
        torch::load(pi_net, pi_path);
    }
    
    this->q = register_module("QNet", q_net);
    this->pi = register_module("PolicyNet", pi_net);
}



float NetDDPRImpl::act(torch::Tensor s)
{
    torch::NoGradGuard no_grad;
    #if DEBUG_LEVEL >= 2
    for(const auto &p: this->q->parameters())
    {
        cout << p.requires_grad() << endl;
    }
    #endif
    return this->pi->forward(s).item<float>();
}

void NetDDPRImpl::save(string pi_path, string q_path)
{
    torch::save(this->pi, pi_path);
    torch::save(this->q, q_path);
}

