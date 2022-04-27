#include "search_modules/strategy_providers/DDPRLabeler.h"

DDPRLabelerOptions::DDPRLabelerOptions(){
    gamma=0.99;
    lr_q=1e-4;
    lr_pi=1e-5 * 0.3;
    polyak=0.995;
    num_epoch=10;
    max_steps=20000;
    update_start_epoch=3;
    buffer_size=int64_t(1e6);
    noise_scale=0.1;
    epsilon = 0.5;
    batch_size=100;
    update_freq=10;
    tail_updates=50;
    max_num_contour=50;
    rnn_hidden_size = 16;
    rnn_num_layers = 1;
    operator_option=DDPRLabeler::OperatorOptions::INFERENCE;
}

DDPRLabeler::DDPRLabeler(int64_t state_dim, int64_t action_dim, Pdd action_range, string load_q_path, string load_pi_path, string q_optim_path, string pi_optim_path, DDPRLabelerOptions options)
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
    srand(2);
    torch::manual_seed(2);

    // set up MLP    
    net = std::make_shared<NetDDPRImpl>(NetDDPROptions({
        .state_dim = state_dim,
        .action_dim = action_dim,
        .action_range = action_range,
        .q_path = load_q_path,
        .pi_path = load_pi_path,
        .max_num_contour = options.max_num_contour,
        .rnn_hidden_size = options.rnn_hidden_size,
        .rnn_num_layers = options.rnn_num_layers
    }));
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
    
    if(load_q_path != "" && load_pi_path != "" && q_optim_path != "" && pi_optim_path != ""){
        torch::load(*optimizer_q, q_optim_path);
        torch::load(*optimizer_pi, pi_optim_path);
    }

    // set up tracking param
    last_action = 0.0;
    step = 0;
    epoch = 0;
    update_count = 0;

    // set up contour candidate vector
    for(int64_t i = 1; i <= action_range.second - 1; i++){
        contour_candidates.push_back(i);
    }

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
    tail_updates = options.tail_updates;
    max_num_contour = options.max_num_contour;
    rnn_hidden_size = options.rnn_hidden_size;
    rnn_num_layers = options.rnn_num_layers;
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

// tuple (prob, noise, floor_label)
std::tuple<float, float, float> DDPRLabeler::train(vector<float> state_flat, vector<float> contour_snapflat, int operator_option)
{
    #if DEBUG_LEVEL >= 2
    cout << "state_flated array: ";
    for(auto it: state_flat)
        cout<<it<<" ";
    cout<<endl;
    #endif
    torch::Tensor tensor_s = torch::from_blob(state_flat.data(), {1, int64_t(state_flat.size())}).clone();    
    // (N: 1, L: max_num_contour, H_{in}: 1)    
    torch::Tensor tensor_contour = torch::from_blob(contour_snapflat.data(), {1, int64_t(contour_snapflat.size()), 1}).clone();  
    float prob;
    float noise;
    float floor_label;

    // forward and get label
    {
        InferenceMode guard(true);
        torch::Tensor out = net->pi->forward(tensor_s, tensor_contour);
        prob = out.index({0, 0}).item<float>();
        noise = out.index({0, 1}).item<float>();
        floor_label = out.index({0, 2}).item<float>();
        #if TORCH_DEBUG >= 0
        cout << "prob: " << prob << " noise: " << noise << " floor_label: " << floor_label << endl;
        #endif
        // Clipping and extending is done in forward function
    }
    // add exploration noise if training
    if(operator_option == OperatorOptions::RANDOM)
    {
        floor_label = (rand() % (int)(action_range.second - 1)) + 1;
        noise = (rand() % 100) / 100.0;
        noise = (rand() % 2) == 0 ? noise : -noise;
        prob = (rand() % 10 > 2) ? ((rand() % 50) / 100.0 + 0.5): ((rand() % 50) / 100.0 + 0.5);
        #if TORCH_DEBUG >= 0
        cout << "RANDOM: " << "floor_label: " << floor_label << " noise: " << noise << " prob: " << prob << endl;
        #endif
    }
    else if(operator_option == OperatorOptions::TRAIN)
    {       
        /* ======================== Exploration design ============================ /
            | 0 <= rd < 0.1 | 0.1 < rd <= 0.2 | 0.2 < rd <= 0.5 | 0.5 <= rd <= 1 | 
                add noise       last action       add int noise     do nothing
        /  ======================================================================== */
        // float random_num = (rand() % 10) / 10.0;         
        // if(random_num < 0.1)
        // {
        //     label += ((rand() % 10) - 5) / 100.0;
        //     cout << "randomed label: " << label << endl;
        // }
        // else if(random_num < 0.2)
        // {
        //     label = last_action;
        // }
        // else if(random_num < 0.4)
        // {
        //     label = contour_candidates[rand() % contour_candidates.size()];
        //     cout << "randomed int label: " << label << endl;
        // }
        
    }     
    return std::make_tuple(prob, noise, floor_label);
}

float DDPRLabeler::label_decision(const ActorOut &in)
{
    const float &floor = std::get<0>(in);
    const float &noise = std::get<1>(in);
    const float &prob = std::get<2>(in);
    assertm("label_decision(): floor is out of range", (floor > 0) && (floor < action_range.second));
    return (prob > 0.5) ? floor : noise + floor;
}

float DDPRLabeler::label_decision(ActorOut &in, bool explore, float epsilon)
{
    if(explore != true)
        throw("label_decision: this function is only for exploration, set the second argument to be true or use the overload with single argument instead.");        
    float &prob = std::get<0>(in);
    float &noise = std::get<1>(in);
    float &floor = std::get<2>(in);
    float label = (prob > 0.5) ? floor : noise + floor;
    assertm("label_decision(): floor is out of range", (floor > 0) && (floor < action_range.second));
    
    // implement epsilon greedy
    if((rand() % 100) / 100.0 < epsilon)
        return label;
    else
    {
        // exlpore
        floor = (rand() % (int)(action_range.second - 1)) + 1;
        assertm("label_decision(): floor is out of range", (floor > 0) && (floor < action_range.second));
        noise = (rand() % 100) / 100.0;
        prob = (rand() % 100) / 100.0;
        label = (prob > 0.5) ? floor : noise + floor;
        return label;
    }    
}


float DDPRLabeler::operator()(vector<float> flatten, vector<float> contour_snapflat, int operator_option)
{
    torch::Tensor tensor_in = torch::from_blob(flatten.data(), {1, int64_t(flatten.size())}).clone();    
    torch::Tensor tensor_contour = torch::from_blob(contour_snapflat.data(), {1, int64_t(contour_snapflat.size()), 1}).clone();  

    float label;
    float prob;
    float noise;
    float floor_label;

    // forward and get label
    {
        InferenceMode guard(true);
        torch::Tensor out = net->pi->forward(tensor_in, tensor_contour);
        prob = out[0].item<float>();
        noise = out[1].item<float>();
        floor_label = out[2].item<float>();  
        // Clipping and extending is done in forward function
    }

    label = label_decision(std::make_tuple(prob, noise, floor_label));
    assertm("label_decision(): label is out of range", (label > 0) && (label < action_range.second));
    return label;
}



torch::Tensor DDPRLabeler::compute_q_loss(const Batch &batch_data)
{   
    const torch::Tensor &s = get<0>(batch_data);
    const torch::Tensor &a = get<1>(batch_data);
    const torch::Tensor &r = get<2>(batch_data);
    const torch::Tensor &s_next = get<3>(batch_data);
    const torch::Tensor &done = get<4>(batch_data);    
    const torch::Tensor &contour_snapshot = get<5>(batch_data); 
    const torch::Tensor &contour_snapshot_next = get<6>(batch_data); 

    torch::Tensor target_qval;
    torch::Tensor raw_target_q;
    torch::Tensor tar_pi;
    {
        torch::NoGradGuard no_grad;
        tar_pi = net_tar->pi->forward(s_next, contour_snapshot_next);
        raw_target_q = net_tar->q->forward(s_next, contour_snapshot_next, tar_pi);
        // raw_target_q = (net_tar->q->forward(s_next, net_tar->pi->forward(s_next)));
        #if TORCH_DEBUG >= 1
        cout << "raw_target_q: " << raw_target_q << endl;
        cout << "~done" << (~done) << endl;
        cout << "r" << r << endl;
        #endif
        target_qval = r + (~done) * gamma * raw_target_q;
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
    torch::Tensor loss = (net->q->forward(s, contour_snapshot, a) - target_qval).pow(2).mean();
    
    if(loss.item<float>() > 1e10 || loss.item<float>() == std::numeric_limits<float>::infinity())
    {
        cout << "s_next: " << s_next << endl;
        cout << "s: " << s << endl;
        cout << "a: " << a << endl;
        cout << "tar_pi: " << tar_pi << endl;
        cout << "loss: " << loss.item<float>() << endl;
        cout << "raw_target_q: " << raw_target_q << endl;
        cout << "target_qval: " << target_qval << endl;
        cout << "qval: " << net->q->forward(s, contour_snapshot, a) << endl;
        cout << "done: " << done << endl;   
        throw("the loss is too large"); 
    }
    return loss;
}

torch::Tensor DDPRLabeler::compute_pi_loss(const Batch &batch_data)
{    
    const torch::Tensor &s = get<0>(batch_data);
    const torch::Tensor &contour_snapshot = get<5>(batch_data);

    torch::Tensor pi_action = net->pi->forward(s, contour_snapshot);
    cout << "pi_action: " << pi_action << endl;
    torch::Tensor q_val = net->q->forward(s, contour_snapshot, pi_action);
    cout << "q_val: " << q_val << endl;
    torch::Tensor loss = -q_val.mean();
    cout << "loss: " << loss << endl;

    // torch::Tensor loss = -(net->q->forward(s, contour_snapshot, net->pi->forward(s, contour_snapshot))).mean();
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

    // update history tracking
    update_count++;
    const int mean_size = 1;
    if(update_count % mean_size == 0)
    {
        float mean_q = std::accumulate(q_loss_vec.begin(), q_loss_vec.end(), 0.0) / mean_size;
        float mean_pi = std::accumulate(pi_loss_vec.begin(), pi_loss_vec.end(), 0.0) / mean_size;
        q_loss_vec.clear();
        pi_loss_vec.clear();
        q_mean_loss.push_back(mean_q);
        pi_mean_loss.push_back(mean_pi);
    }


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