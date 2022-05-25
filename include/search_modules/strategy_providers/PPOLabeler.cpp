#include "search_modules/strategy_providers/PPOLabeler.h"

namespace PPO{

PPOLabeler::PPOLabeler(PPOLabelerOptions opt)
: opt(opt)
{       
    // Debug
    cout << "state_dim: " << opt.state_dim() << endl;
    cout << "action_dim: " << opt.action_dim() << endl;

    // set up buffer
    buffer = std::make_shared<PPO::ReplayBufferImpl>(opt.buffer_size());

    // set up MLP    
    net = std::make_shared<NetPPOImpl>(NetPPOOptions({
        .state_dim = opt.state_dim(),
        .action_dim = opt.action_dim(),        
        .q_path = opt.load_q_path(),
        .pi_path = opt.load_pi_path()
    }));

    #if TORCH_DEBUG >= -1
    cout << "weight of original net: " << endl;
    for(auto &param : net->named_parameters())
        cout << param.key() << ": " << param.value() << endl;
    #endif

    // set up optimizer    
    optimizer_q = std::make_shared<torch::optim::Adam>(net->q->parameters(), opt.lr_q());
    optimizer_pi = std::make_shared<torch::optim::Adam>(net->pi->parameters(), opt.lr_pi());    
    
    if(opt.load_q_path() != "" && opt.load_pi_path() != "" && opt.q_optim_path() != "" && opt.pi_optim_path() != ""){
        torch::load(*optimizer_q, opt.q_optim_path());
        torch::load(*optimizer_pi, opt.pi_optim_path());
    }

    // set up tracking param    
    _step = 0;
    _update_count = 0;
    _epoch = 0;    

    // set up labeler state
    labeler_state = LabelerState::TRAIN_RUNNING;
}

PPOLabeler::LabelerState PPOLabeler::get_labeler_state()
{
    // Inferencing
    if(labeler_state == LabelerState::INFERENCE)
    {
        labeler_state = LabelerState::INFERENCE;
        return LabelerState::INFERENCE;
    }
    // Training 
    if(_epoch < opt.num_epoch() && _step < opt.steps_per_epoch()){
        labeler_state = LabelerState::TRAIN_RUNNING;
        return LabelerState::TRAIN_RUNNING;
    }
    else{
        labeler_state = LabelerState::TRAIN_EPOCH_END;
        return LabelerState::TRAIN_EPOCH_END;
    }
}

// Do training and buffering here, return the label if created, otherwise go throgh the network again
float PPOLabeler::operator()(vector<float> state_flat)
{    
    // get current labeler_state
    PPOLabeler::LabelerState labeler_state_ = this->get_labeler_state();

    if(labeler_state == LabelerState::INFERENCE)
        return 0;
    if(labeler_state == LabelerState::TRAIN_EPOCH_END)
        return 0;
    
    torch::Tensor tensor_s = torch::from_blob(state_flat.data(), {1, int64_t(state_flat.size())}).clone();        

    // update buffer (s, a, '', v, logp) -> (s, a, r, v, logp)
    buffer->prep.r() = node_reward;        
    if(this->buffer->safe_to_submit())
    {
        this->buffer->submit();            
    }
    else
    {
        assertm("unsafe submission", false);
    }

    if(labeler_state == LabelerState::TRAIN_RUNNING)
    {        
        PPO::StepOutput out = net->step(tensor_s);
        auto &[action, val, logp] = std::tie(out.a, out.v, out.logp);

        // update buffer () -> (s, a, '', v, logp)
        std::tie(buffer->prep.s(), buffer->prep.a(), buffer->prep.val(), buffer->prep.logp()) = std::tie(state_flat, action, val, logp);

        // increase step
        this->step()++;

        while(action != PPO::ACTIONS::PLACE && action != PPO::ACTIONS::PLACE_INSERT)
        {        
            // update buffer (s, a, '', v, logp) -> (s, a, r, v, logp)
            buffer->prep.r() = move_reward;
            if(buffer->safe_to_submit())
            {
                buffer->submit();
            }

            // update contour pointer 
            if(action == PPO::ACTIONS::LEFT)
                this->action_left();
            else if(action == PPO::ACTIONS::RIGHT)
                this->action_right();
            else
            {
                assertm("Invalid action", false);
            }

            // get state
            STATE_ENCODING state_flat = this->get_state();
            torch::Tensor tensor_s = torch::from_blob(state_flat.data(), {1, int64_t(state_flat.size())}).clone();        

            PPO::StepOutput out = net->step(tensor_s);
            std::tie(action, val, logp) = std::tie(out.a, out.v, out.logp);

            // update buffer () -> (s, a, '', v, logp) 
            std::tie(buffer->prep.s(), buffer->prep.a(), buffer->prep.val(), buffer->prep.logp()) = std::tie(state_flat, action, val, logp);

            // increase step
            this->step()++;
        }

        /* ==== ACTION be PLACE or PLACE_INSERT START ==== */
        /* */
        assertm("Logical invalid action", action == PPO::ACTIONS::PLACE || action == PPO::ACTIONS::PLACE_INSERT);

        // execute action
        if(action == PPO::ACTIONS::PLACE)
            this->action_place();
        else if(action == PPO::ACTIONS::PLACE_INSERT)
            this->action_place_insert();
        else
        {
            assertm("Invalid action", false);
        }

        // interpret current state to label
        label = this->interpret_state();

        /* */
        /* ==== ACTION be PLACE or PLACE_INSERT END ==== */

        return label;

    }
    else if(labeler_state == LabelerState::INFERENCE)
    {
        // increase step
        PPO::StepOutput out = net->step(tensor_s);
        auto &[action, val, logp] = std::tie(out.a, out.v, out.logp);
        while(action != PPO::ACTIONS::PLACE && action != PPO::ACTIONS::PLACE_INSERT)
        {
            // update contour pointer 
            if(action == PPO::ACTIONS::LEFT)
                this->action_left();
            else if(action == PPO::ACTIONS::RIGHT)
                this->action_right();
            else
            {
                assertm("Invalid action", false);
            }

            // get state
            STATE_ENCODING state_flat = this->get_state();
            torch::Tensor tensor_s = torch::from_blob(state_flat.data(), {1, int64_t(state_flat.size())}).clone();        

            // keep going        
            out = net->step(tensor_s);
            std::tie(action, val, logp) = std::tie(out.a, out.v, out.logp);
        }
        if(action == PPO::ACTIONS::PLACE)
            this->action_place();
        else if(action == PPO::ACTIONS::PLACE_INSERT)
            this->action_place_insert();
        else
        {
            assertm("Invalid action", false);
        }
        float label = this->interpret_state();
        return label;
    }

}

//                 s                a            r                s'            adv,        logp          //    
// typedef tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Batch;
tuple<torch::Tensor, PPO::ExtraInfo> PPOLabeler::compute_pi_loss(const PPO::Batch &batch_data)
{   
    const torch::Tensor &s = batch_data.s;
    const torch::Tensor &a = batch_data.a;
    const torch::Tensor &r = batch_data.r;
    const torch::Tensor &adv = batch_data.adv;
    const torch::Tensor &logp_old = batch_data.logp;

    torch::Tensor logp = net->pi->forward(s, a);
    torch::Tensor importance_weight = torch::exp(logp - logp_old);
    torch::Tensor clip_adv = torch::clamp(importance_weight, 1-opt.clip_ratio(), 1+opt.clip_ratio()) * adv;
    torch::Tensor loss = -torch::min(importance_weight * adv, clip_adv).mean();

    // ExtraInfo
    torch::Tensor approx_kl = (logp_old - logp).mean();
    torch::Tensor entropy /*= (torch::exp(logp) + torch::exp(-logp)).mean()*/;
    torch::Tensor clipfrac = (torch::logical_or(importance_weight.gt(1+opt.clip_ratio()), importance_weight.lt(1-opt.clip_ratio()))).mean();
    PPO::ExtraInfo extra_info = {
        .approx_kl = approx_kl.item<float>(),
        .entropy = entropy.item<float>(),
        .clipfrac = clipfrac.item<float>()
    };

    if(loss.item<float>() > 1e10 || loss.item<float>() == std::numeric_limits<float>::infinity())
    {
        cout << "s: " << s << endl;
        cout << "a: " << a << endl;
        cout << "loss: " << loss.item<float>() << endl;
        throw("the loss is too large"); 
    }
    return std::make_tuple(loss, extra_info);
}

torch::Tensor PPOLabeler::compute_q_loss(const PPO::Batch &batch_data)
{    
    const torch::Tensor &s = batch_data.s;
    const torch::Tensor &r = batch_data.r;
    torch::Tensor v = net->q->forward(s);
    torch::Tensor v_loss = (r - v).pow(2).mean();
    return v_loss;
}

void PPOLabeler::update(const PPO::SampleBatch &batch_data)
{       
    using std::cout, std::endl; 
    const int mean_size = 1;
    #if TORCH_DEBUG >= -1
    cout << "before update" << endl;
    cout << "-------------" << endl;
    layer_weight_print(*(net->q));
    layer_weight_print(*(net->pi));
    cout << "-------------" << endl << endl;
    #endif
    
    PPO::Batch batch = buffer->getBatchTensor(batch_data);

    // prepare loss for history
    torch::Tensor pi_loss_old, v_loss_old;
    PPO::ExtraInfo info;
    std::tie(pi_loss_old, info) = compute_pi_loss(batch);
    v_loss_old = compute_q_loss(batch);
    float pi_loss_old_val = pi_loss_old.item<float>();
    float v_loss_old_val = v_loss_old.item<float>();
    q_loss_vec.push_back(v_loss_old_val);
    pi_loss_vec.push_back(pi_loss_old_val);
    
    for(int64_t i = 0; i < opt.train_pi_iter(); i++)
    {
        optimizer_pi->zero_grad();
        torch::Tensor pi_loss;
        PPO::ExtraInfo info;
        std::tie(pi_loss, info) = compute_pi_loss(batch);
        float approx_kl = info.approx_kl;
        if(approx_kl > 1.5 * opt.target_kl())
        {
            cerr << "Early stopping at step " << this->_step << "due to reaching max kl." << endl;
            break;
        }
        pi_loss.backward();
        optimizer_pi->step();
    }

    for(int64_t i = 0; i < opt.train_q_iter(); i++)
    {
        optimizer_q->zero_grad();
        torch::Tensor v_loss = compute_q_loss(batch);
        v_loss.backward();
        optimizer_q->step();
    }    

    this->_update_count++;
}

}; // namespace PPO