#include "search_modules/Net/DDPR/NetDDPR.h"

ReplayBufferImpl::ReplayBufferImpl(int max_size) {
    this->max_size = max_size;
    this->idx = 0;
    this->size = 0;
    s.resize(max_size);
    a.resize(max_size);
    r.resize(max_size);
    s_next.resize(max_size);
    done.resize(max_size);
}

void ReplayBufferImpl::push(vector<float> s, float a, int r, vector<float> s_, bool done)
{
    if(idx >= max_size)
    {
        cout << "replay buffer index out of bound" << endl;
        exit(LOGIC_ERROR);
    }
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

vector<float> StateInput::flatten_and_norm()
{   
    
}


// should output in tensor form
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ReplayBufferImpl::get(vector<int> indecies)
{
    int batch_size = indecies.size();
    int state_feature_size = s[0].size();
    vector<vector<float>> s;
    vector<float> a;
    vector<float> r;
    vector<vector<float>> s_next;
    vector<float> done;
    for(int i = 0; i < indecies.size(); i++)
    {
        int idx = indecies[i];
        s.push_back(s[idx]);
        a.push_back(a[idx]);
        r.push_back(r[idx]);
        s_next.push_back(s_next[idx]);
        done.push_back(done[idx]);
    }
    // flatten the state array
    vector<float> s_flat;
    vector<float> s_next_flat;
    for(int i = 0; i < batch_size; i++)
    {
        s_flat.insert(s_flat.end(), make_move_iterator(s[i].begin()), make_move_iterator(s[i].end()));
        s_next_flat.insert(s_next_flat.end(), make_move_iterator(s_next[i].begin()), make_move_iterator(s_next[i].end()));
    }

    // turn arrays to Tensor
    Tensor s_tensor = torch::from_blob(s_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor a_tensor = torch::from_blob(a.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor r_tensor = torch::from_blob(r.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor s_next_tensor = torch::from_blob(s_next_flat.data(), {batch_size, state_feature_size}, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor done_tensor = torch::from_blob(done.data(), {batch_size, 1}, torch::TensorOptions().dtype(torch::kBool));

    return std::make_tuple(s_tensor, a_tensor, r_tensor, s_next_tensor, done_tensor);
}


NetDDPRImpl::NetDDPRImpl(int64_t state_dim, int64_t action_dim, Pdd action_range)
{
    this->state_dim = state_dim;
    this->action_dim = action_dim;
    this->action_range = action_range;

    this->q = register_module("QNet", NetDDPRQNet(state_dim, action_dim));
    this->pi = register_module("PolicyNet", NetDDPRActor(state_dim, action_range));
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

