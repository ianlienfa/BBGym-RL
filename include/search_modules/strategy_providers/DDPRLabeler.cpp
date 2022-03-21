#include "search_modules/strategy_providers/DDPRLabeler.h"

DDPRLabeler::DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, double gamma=0.99, double lr_q=1e-3, double lr_pi=1e-3, double polyak=0.995, int64_t num_epoch=150, int64_t max_steps=1000, int64_t update_start_epoch=10, int64_t buffer_size=int64_t(1e6), double noise_scale=0.1, int64_t batch_size=100, int64_t update_freq=5)
{       
    // set up buffer
    buffer = ReplayBuffer(buffer_size);

    // set up random seeds
    srand(0);

    // set up MLP
    net = NetDDPR(state_dim, action_dim, action_range);
    net_tar = net.clone().detach();
    for (auto& p : net_tar->parameters())
        p.requires_grad = false;

    // set up optimizer
    optimizer_q = torch::optim::Adam(net->q->parameters(), lr_q);
    optimizer_pi = torch::optim::Adam(net->pi->parameters(), lr_pi);    
}

// DDPRLabeler::operator()(const OneRjSumCjNode &node) const
// {
//     // get state
//     if(s != NULL)
//         s_next = StateInput(node);
//     else    
//         s = StateInput(node);
    
//     // buffer data collection (s, label, r, done should be given value at the last time point)
//     buffer.push(s, label, r, s_next, done);
//     torch::Tensor tensor_in = torch::from_blob(s.flatten()).clone();    
//     label = ddpg(tensor_in);    
//     return label;
// }

DDPRLabeler::operator()(const StateInput &input) const
{        
    torch::Tensor tensor_in = torch::from_blob(input.flatten_and_norm()).clone();    
    
    // forward and get label
    {
        InferenceMode guard(true);
        label = net.pi(tensor_in);
    }
    return label;
}


void DDPRLabeler::fill_response(float reward, bool done)
{
    this->r = reward;
    this->done = done;
}

torch::Tensor compute_q_loss(const Batch &batch_data)
{   
    const torch::Tensor &s = get<0>(batch_data);
    const torch::Tensor &a = get<1>(batch_data);
    const torch::Tensor &r = get<2>(batch_data);
    const torch::Tensor &s_next = get<3>(batch_data);
    const torch::Tensor &done = get<4>(batch_data);
    torch::Tensor target_qval;
    {
        torch::NoGradGuard no_grad;
        target_qval = r + this->gamma * (1 - done) * net_tar->q(s_next, net_tar->pi(s_next));        
        // Is this the right way to do it?
    }
    torch::Tensor loss = (net->q(s, a) - target_qval).pow(2).mean();
    return loss;
}

torch::Tensor compute_pi_loss(const Batch &batch_data)
{
    const torch::Tensor &s = get<0>(batch_data);
    torch::Tensor loss = -(net->q(s, net->pi(s))).mean();
    return loss;
}

void DDPRLabeler::update(Batch &batch_data)
{
    // compute loss
    torch::Tensor q_loss = compute_q_loss(batch_data);
    torch::Tensor pi_loss = compute_pi_loss(batch_data);

    // update
    optimizer_q.zero_grad();
    q_loss.backward();
    optimizer_q.step();

    optimizer_pi.zero_grad();
    pi_loss.backward();
    optimizer_pi.step();
}


// do actual training here
int DDPRLabeler::ddpg_train(Batch &batch_data)
{   

    // update q  
    optimizer_q.zero_grad();
    torch::Tensor q_loss = compute_q_loss(batch_data);
    q_loss.backward();
    optimizer_q.step();

    // stablize the gradient for q-network
    for(auto& p : net->q->parameters())
        p.requires_grad_(false);

    // update pi
    optimizer_pi.zero_grad();        
    torch::Tensor pi_loss = compute_pi_loss(batch_data);    
    pi_loss.backward();
    optimizer_pi.step();

    // resume gradient computation for q-network
    for(auto& p : net->q->parameters())
        p.requires_grad_(true);

    {
        // update target network
        torch::NoGradGuard no_grad;
        for (auto& p = net->parameters().begin(), p_tar = net_tar->parameters().begin(); p != net->parameters().end(); ++p, ++p_tar)
        {
            (*p_tar).data().mul_(this->polyak);
            (*p_tar).data().add_((1 - this->polyak) * (*p_tar));
        }
    }

    // set up result meomization
    throw NotImplemented;
    this->step++;
    vector<float> test_loss;
    vector<float> test_epoch;    

}
