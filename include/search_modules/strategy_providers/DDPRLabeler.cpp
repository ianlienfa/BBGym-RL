#include "search_modules/strategy_providers/DDPRLabeler.h"

DDPRLabelerOptions::DDPRLabelerOptions(){
    gamma=0.99;
    lr_q=1e-3;
    lr_pi=1e-2;
    polyak=0.995;
    num_epoch=150;
    max_steps=20000;
    update_start_epoch=10;
    buffer_size=int64_t(1e6);
    noise_scale=0.1;
    epsilon = 0.5;
    batch_size=100;
    update_freq=50;
    operator_option=DDPRLabeler::OperatorOptions::INFERENCE;
}

DDPRLabeler::DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, string load_q_path, string load_pi_path, DDPRLabelerOptions options)
: state_dim(state_dim), action_dim(action_dim), action_range(action_range)
{       

    // Debug
    cout << "state_dim: " << state_dim << endl;
    cout << "action_dim: " << action_dim << endl;
    cout << "action_range: " << action_range.first << " " << action_range.second << endl;

    // parameters init
    fill_option(options);

    // set up buffer
    buffer = std::make_shared<ReplayBufferImpl>(buffer_size);

    // set up random seeds
    srand(time(NULL));
    torch::manual_seed(time(NULL));

    // set up MLP
    net = NetDDPR(state_dim, action_dim, action_range, load_q_path, load_pi_path);
    auto net_copy_ptr = net->clone();
    net_tar = std::dynamic_pointer_cast<NetDDPRImpl>(net_copy_ptr);

    #if TORCH_DEBUG >= 1
    cout << "weight of copied net: " << endl;
    layer_weight_print((*net_copy_ptr));
    
    cout << "weight of original net: " << endl;
    for(auto &param : net->named_parameters())
        cout << param.key() << ": " << param.value() << endl;
    #endif

    for (auto& p : net_tar->parameters())
        p.detach_();

    // set up optimizer
    optimizer_q = std::make_shared<torch::optim::Adam>(net->q->parameters(), lr_q);
    optimizer_pi = std::make_shared<torch::optim::Adam>(net->pi->parameters(), lr_pi);    

    // set up tracking param
    last_action = 0.0;
    step = 0;
    epoch = 0;

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
void DDPRLabeler::fill_option(const DDPRLabelerOptions &options)
{
    gamma = options.gamma;
    lr_q = options.lr_q;
    lr_pi = options.lr_pi;
    polyak = options.polyak;
    noise_scale = options.noise_scale;
    num_epoch = options.num_epoch;
    max_steps = options.max_steps;
    update_start_epoch = options.update_start_epoch;
    buffer_size = options.buffer_size;
    batch_size = options.batch_size;
    update_freq = options.update_freq;
    epsilon = options.epsilon;
}


// float DDPRLabeler::operator()(StateInput input) 
// {        
//     vector<float> flatten = input.get_state_encoding();
//     torch::Tensor tensor_in = torch::from_blob(flatten.data(), {1, int64_t(flatten.size())}).clone();    
//     float label;
//     // forward and get label
//     {
//         InferenceMode guard(true);
//         label = net->pi->forward(tensor_in).item<float>();
//     }
//     return label;
// }


float DDPRLabeler::operator()(vector<float> flatten, int operator_option)
{
    #if DEBUG_LEVEL >= 2
    cout << "flattened array: ";
    for(auto it: flatten)
        cout<<it<<" ";
    cout<<endl;
    #endif
    torch::Tensor tensor_in = torch::from_blob(flatten.data(), {1, int64_t(flatten.size())}).clone();    
    float label;
    // forward and get label
    {
        InferenceMode guard(true);
        label = net->pi->forward(tensor_in).item<float>();  
        // Clipping and extending is done in forward function
    }
    // add exploration noise if training
    if(operator_option == OperatorOptions::RANDOM)
    {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);  // 0.0 ~ 1.0
        label = r * action_range.second; // 0.0 ~ action_range.second
        label = ((rand() % 10) / 10.0 < epsilon) ? floor(label) : last_action;  // add some prob to go to same contour
    }
    else if(operator_option == OperatorOptions::TRAIN)
    {       
        float random_num = (rand() % 10) / 10.0; 
        if(random_num < epsilon)
        {
            // do last action
            if(random_num < epsilon / 2)
            {
                label = last_action;
            }
            // floor to nearest contour
            else
            {
                /* 
                    The rounding should be floor() instead of round()
                    to preserve the extendibility of the action space.
                */
                label = floor(label);                
            }
        }
    }    
    last_action = label;
    return label;
}



torch::Tensor DDPRLabeler::compute_q_loss(const Batch &batch_data)
{   
    const torch::Tensor &s = get<0>(batch_data);
    const torch::Tensor &a = get<1>(batch_data);
    const torch::Tensor &r = get<2>(batch_data);
    const torch::Tensor &s_next = get<3>(batch_data);
    const torch::Tensor &done = get<4>(batch_data);    
    torch::Tensor target_qval;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor raw_target_q = (net_tar->q->forward(s_next, net_tar->pi->forward(s_next)));
        cout << "raw_target_q: " << raw_target_q << endl;
        cout << "~done" << done << endl;
        cout << "r" << r << endl;
        target_qval = r + (~done) * gamma * raw_target_q;
        // target_qval = r + this->gamma * ~(done) * (net_tar->q->forward(s_next, net_tar->pi->forward(s_next)));        
        // Is this the right way to do it?
    }
    // cout << "target_qval: " << target_qval << endl;
    // cout << "a: " << a << endl;
    // torch::Tensor qval = net->q->forward(s, a);
    // cout << "qval: " << qval << endl;    
    // torch::Tensor loss = qval - target_qval; // diff
    // cout << "loss_diff: " << loss << endl;
    // loss = loss.pow(2);
    // cout << "loss_pow2: " << loss << endl;
    // loss = loss.mean();
    // cout << "loss_mean: " << loss << endl;
    torch::Tensor loss = (net->q->forward(s, a) - target_qval).pow(2).mean();    
    return loss;
}

torch::Tensor DDPRLabeler::compute_pi_loss(const Batch &batch_data)
{    
    const torch::Tensor &s = get<0>(batch_data);
    torch::Tensor loss = -(net->q->forward(s, net->pi->forward(s))).mean();
    return loss;
}

void DDPRLabeler::update(const RawBatch &batch_data)
{       
    using std::cout, std::endl; 
    #if TORCH_DEBUG >= 1
    cout << "before update" << endl;
    cout << "-------------" << endl;
    layer_weight_print(*(net->q));
    layer_weight_print(*(net->pi));
    cout << "-------------" << endl << endl;
    #endif

    
    Batch batch = buffer->getBatchTensor(batch_data);

    // update q  
    optimizer_q->zero_grad();
    torch::Tensor q_loss = compute_q_loss(batch);
    #if TORCH_DEBUG >= 1
    cout << "action tensor after q update" << endl;
    cout << get<1>(batch) << endl;
    cout << "q_loss: " << q_loss << endl;    
    #endif
    q_loss_vec.push_back(q_loss.item<float>());
    q_loss.backward();
    optimizer_q->step();

    #if TORCH_DEBUG >= 1
    cout << "updated q" << endl;
    cout << "-------------" << endl;
    layer_weight_print(*(net->q));
    cout << "-------------" << endl << endl;
    #endif

    // stablize the gradient for q-network
    for(auto& p : net->q->parameters())
        p.requires_grad_(false);

    // update pi
    optimizer_pi->zero_grad();       
    torch::Tensor pi_loss = compute_pi_loss(batch);
    pi_loss_vec.push_back(pi_loss.item<float>());
    #if TORCH_DEBUG >= 1
    cout << "action tensor after pi update" << endl;
    cout << get<1>(batch) << endl;
    #endif
    pi_loss.backward();
    optimizer_pi->step();

    #if TORCH_DEBUG >= 1
    cout << "updated pi" << endl;
    cout << "-------------" << endl;
    layer_weight_print(*(net->pi));
    cout << "-------------" << endl << endl;
    #endif

    // resume gradient computation for q-network
    for(auto& p : net->q->parameters())
        p.requires_grad_(true);
    
    {
        // update target network
        torch::NoGradGuard no_grad;
        #if TORCH_DEBUG >= 1
        cout << "no grad" << endl;
        cout << "-------------" << endl;
        layer_weight_print(*(net));
        cout << "-------------" << endl << endl;
        #endif    

        // update target network, set recurse=true to update all layers
        auto param_pi = net->pi->parameters(true);
        auto param_q = net->q->parameters(true);
        auto param_pi_tar = net_tar->pi->parameters(true);
        auto param_q_tar = net_tar->q->parameters(true);

        auto p_iter = param_pi.begin();
        auto p_tar_iter = param_pi_tar.begin();
        while (p_iter != param_pi.end())
        {
            #if VALIDATION_LEVEL == validation_level_HIGH
            if ((*p_iter).sizes() != (*p_tar_iter).sizes())
            {
                cout << "p_iter: " << (*p_iter).sizes() << endl;
                cout << "p_tar_iter: " << (*p_tar_iter).sizes() << endl;
                throw std::runtime_error("parameter size mismatch");
            }            
            #endif
       
            (*p_tar_iter).mul_(this->polyak);            
            (*p_tar_iter).add_((1 - this->polyak) * (*p_iter));

            ++p_iter;
            ++p_tar_iter;
        }

        p_iter = param_q.begin();
        p_tar_iter = param_q_tar.begin();
        while (p_iter != param_q.end())
        {
            #if VALIDATION_LEVEL == validation_level_HIGH
            if ((*p_iter).sizes() != (*p_tar_iter).sizes())
            {
                cout << "p_iter: " << (*p_iter).sizes() << endl;
                cout << "p_tar_iter: " << (*p_tar_iter).sizes() << endl;
                throw std::runtime_error("parameter size mismatch");
            }            
            #endif
            (*p_tar_iter).mul_(this->polyak);            
            (*p_tar_iter).add_((1 - this->polyak) * (*p_iter));

            ++p_iter;
            ++p_tar_iter;
        }
    }
}


// do actual training here
// int DDPRLabeler::ddpg_train(Batch &batch_data)
// {   

//     // update q  
//     optimizer_q->zero_grad();
//     torch::Tensor q_loss = compute_q_loss(batch_data);
//     q_loss.backward();
//     optimizer_q->step();

//     // stablize the gradient for q-network
//     for(auto& p : net->q->parameters())
//         p.requires_grad_(false);

//     // update pi
//     optimizer_pi->zero_grad();        
//     torch::Tensor pi_loss = compute_pi_loss(batch_data);    
//     pi_loss.backward();
//     optimizer_pi->step();

//     // resume gradient computation for q-network
//     for(auto& p : net->q->parameters())
//         p.requires_grad_(true);

//     {
//         // update target network
//         torch::NoGradGuard no_grad;
//         for (auto& p = net->parameters().begin(), p_tar = net_tar->parameters().begin(); p != net->parameters().end(); ++p, ++p_tar)
//         {
//             (*p_tar).data().mul_(this->polyak);
//             (*p_tar).data().add_((1 - this->polyak) * (*p_tar));
//         }
//     }

//     // set up result meomization
//     throw NotImplemented;
//     this->step++;
//     vector<float> test_loss;
//     vector<float> test_epoch;    

// }
