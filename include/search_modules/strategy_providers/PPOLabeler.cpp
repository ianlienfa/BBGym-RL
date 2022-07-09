#include "search_modules/strategy_providers/PPOLabeler.h"
namespace PPO{
PPOLabeler::PPOLabeler(PPO::PPOLabelerOptions options) :
    opt(options)
{       
    // Debug
    cout << "state_dim: " << opt.state_dim() << endl;
    cout << "action_dim: " << opt.action_dim() << endl;

    // set up buffer
    buffer = std::make_shared<PPO::ReplayBufferImpl>(opt.buffer_size(), opt.max_num_contour());

    // set up MLP    
    net = std::make_shared<NetPPOImpl>(NetPPOOptions({
        .state_dim = opt.state_dim(),
        .action_dim = opt.action_dim(),    
        .hidden_dim = opt.hidden_dim(),    
        .q_path = opt.load_q_path(),
        .pi_path = opt.load_pi_path()
    }));

    // set undeterministic
    at::globalContext().setDeterministicCuDNN(true);

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
    if(_epoch <= opt.num_epoch() && _step < opt.steps_per_epoch()){
        labeler_state = LabelerState::TRAIN_RUNNING;
        return LabelerState::TRAIN_RUNNING;
    }
    else{
        cout << "Train epoch end assigned: _epoch / num_epoch: " << _epoch << " / " << opt.num_epoch() << " _step / steps_per_epoch : " << _step << " / " << opt.steps_per_epoch() << endl;
        labeler_state = LabelerState::TRAIN_EPOCH_END;
        return LabelerState::TRAIN_EPOCH_END;
    }
}


// Do training and buffering here, return the label if created, otherwise go throgh the network again
int64_t PPOLabeler::operator()(::OneRjSumCjNode& node, vector<float>& state_flat, ::OneRjSumCjGraph& graph)
{        
    // get current labeler_state
    CONTOUR_TYPE label;
    static PPOLabeler::LabelerState labeler_state_ = PPOLabeler::LabelerState::UNDEFINED;
    PPOLabeler::LabelerState current_labeler_state = this->get_labeler_state();
    if(labeler_state_ != current_labeler_state)
    {
        switch (current_labeler_state)    
        {
            case LabelerState::TRAIN_EPOCH_END:
                cerr << "TRAIN_EPOCH_END entered" << endl;
                break;
            case LabelerState::TRAIN_RUNNING:
                cerr << "TRAIN_RUNNING entered" << endl;
                break;
            case LabelerState::INFERENCE:
                cerr << "INFERENCE entered" << endl;
                break;
            
            default:
                cerr << "Undefined state entered" << endl;
                assert(false);
                break;
        }
    }
    labeler_state_ = current_labeler_state;

    // get current state
    torch::Tensor tensor_s = torch::from_blob(state_flat.data(), {1, int64_t(state_flat.size())}).clone();        

    // update buffer (s, a, '', v, logp) -> (s, a, r, v, logp)
    if(!buffer->prep.empty()) // if not the first round, for the first round, ignore the r
    {
        buffer->prep.r() = graph.get_node_reward();      
    }
    if(this->buffer->safe_to_submit())
    {
        this->buffer->submit();            
    }


    if(labeler_state_ == LabelerState::TRAIN_EPOCH_END)
    {
        cout << "Train epoch end operator called" << endl;
        // step == step_per_epoch, time out
        PPO::StepOutput out = net->step(tensor_s);
        auto &val = out.v;
        buffer->finish_epoch(val);
        return 0;
    }
    else if(labeler_state_ == LabelerState::TRAIN_RUNNING)
    {        
        PPO::StepOutput out = net->step(tensor_s);
        auto &action = out.a;
        auto &val = out.v;
        auto &logp = out.logp;
        auto &encoded_a = out.encoded_a;

        // update buffer () -> (s, a, '', v, logp)
        buffer->prep.s() = state_flat;
        buffer->prep.a() = encoded_a;
        buffer->prep.val() = val;
        buffer->prep.logp() = logp;

        // increase step
        this->step()++;

        while(action != PPO::ACTIONS::PLACE && action != PPO::ACTIONS::PLACE_INSERT)
        {        
            // update buffer (s, a, '', v, logp) -> (s, a, r, v, logp)
            buffer->prep.r() = graph.contours.get_picker_reward();
            if(buffer->safe_to_submit())
            {
                buffer->submit();
            }

            // update contour pointer 
            if(action == PPO::ACTIONS::LEFT)
                graph.contours.left();
            else if(action == PPO::ACTIONS::RIGHT)
                graph.contours.right();
            else
            {
                assertm("Invalid action", false);
            }

            // get state
            STATE_ENCODING tweaked_state = PPO::StateInput::get_state_encoding_fast(state_flat, graph);
            #if GAME_TRACKER == 1
            cout << "state encoding: " << tweaked_state << endl;
            #endif
            torch::Tensor tensor_s = torch::from_blob(tweaked_state.data(), {1, int64_t(tweaked_state.size())}).clone();        

            PPO::StepOutput out = net->step(tensor_s);            
            std::tie(action, encoded_a, val, logp) = std::tie(out.a, out.encoded_a, out.v, out.logp);

            // update buffer () -> (s, a, '', v, logp) 
            std::tie(buffer->prep.s(), buffer->prep.a(), buffer->prep.val(), buffer->prep.logp()) = std::tie(tweaked_state, encoded_a, val, logp);

            // increase step
            this->step()++;
        }

        /* ==== ACTION be PLACE or PLACE_INSERT START ==== */
        /* */
        assertm("Logical invalid action", action == PPO::ACTIONS::PLACE || action == PPO::ACTIONS::PLACE_INSERT);


        /* */
        /* ==== ACTION be PLACE or PLACE_INSERT END ==== */
        // execute action
        if(action == PPO::ACTIONS::PLACE)
        {
            label = graph.contours.place(node);
            return label;
        }
        else if(action == PPO::ACTIONS::PLACE_INSERT)
        {
            label = graph.contours.insert_and_place(node);
            return label;
        }
        else
        {
            assertm("Invalid action", false);            
        }             
        return 0;

    }
    else if(labeler_state_ == LabelerState::INFERENCE)
    {
        // increase step
        PPO::StepOutput out = net->step(tensor_s, true /*deterministic*/);
        auto &action = out.a;
        auto &val = out.v;
        auto &logp = out.logp;

        while(action != PPO::ACTIONS::PLACE && action != PPO::ACTIONS::PLACE_INSERT)
        {
            // update contour pointer 
            if(action == PPO::ACTIONS::LEFT)
                graph.contours.left();
            else if(action == PPO::ACTIONS::RIGHT)
                graph.contours.right();
            else
            {
                assertm("Invalid action", false);
            }

            // get state
            STATE_ENCODING tweaked_state = PPO::StateInput::get_state_encoding_fast(state_flat, graph);
            torch::Tensor tensor_s = torch::from_blob(tweaked_state.data(), {1, int64_t(tweaked_state.size())}).clone();        

            // keep going        
            out = net->step(tensor_s);
            std::tie(action, val, logp) = std::tie(out.a, out.v, out.logp);
        }
        
        // execute action
        if(action == PPO::ACTIONS::PLACE)
        {
            label = graph.contours.place(node);
            return label;
        }
        else if(action == PPO::ACTIONS::PLACE_INSERT)
        {
            label = graph.contours.insert_and_place(node);
            return label;
        }
        else
        {
            assertm("Invalid action", false);            
        }             
        return 0;
    }
    assertm("Error control path", false);
    return 0;
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

    // layer_weight_print(*(net->pi));
    torch::Tensor logp = net->pi->forward(s, a);
    if(logp.isnan().any().item<bool>())
    {        
        cerr << "s: " << s << endl;
        cerr << "a: " << a << endl;    
        cerr << "logp_old: " << logp_old << std::endl;
        cerr << "logp: " << logp << std::endl;
        assertm("logp is nan", false);
    }
    torch::Tensor importance_weight = torch::exp(logp - logp_old);
    torch::Tensor clip_adv = torch::clamp(importance_weight, 1-opt.clip_ratio(), 1+opt.clip_ratio()) * adv;
    torch::Tensor loss = -torch::min(importance_weight * adv, clip_adv).mean();

    // ExtraInfo
    torch::Tensor approx_kl = (logp_old - logp).mean();
    torch::Tensor entropy = (torch::exp(logp) + torch::exp(-logp)).mean();
    torch::Tensor clipfrac = (torch::logical_or(importance_weight.gt(1+opt.clip_ratio()), importance_weight.lt(1-opt.clip_ratio()))).toType(kFloat32).mean(0);
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

    // add entropy to loss!
    loss = loss + entropy * opt.entropy_lambda();
    return std::make_tuple(loss, extra_info);
}

torch::Tensor PPOLabeler::compute_q_loss(const PPO::Batch &batch_data)
{    
    const torch::Tensor &s = batch_data.s;
    const torch::Tensor &r = batch_data.r;
    torch::Tensor v = net->q->forward(s);
    torch::Tensor v_loss = (r - v).pow(2).mean();
    if(v_loss.item<float>() > 1e8)
    {
        layer_weight_print(*(net->q));
        cerr << "s: " << s << endl;
        cerr << "r: " << r << endl;
        cerr << "v: " << v << endl;
        cerr << "v_loss: " << v_loss.item<float>() << endl;
    }
    assertm("v_loss too big", v_loss.item<float>() < 1e8);
    return v_loss;
}

void PPOLabeler::update(PPO::SampleBatch &batch_data)
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

    cout << "pi_loss_old: " << pi_loss_old_val << endl;
    cout << "v_loss_old: " << v_loss_old_val << endl;

    q_loss_vec.push_back(v_loss_old_val);
    pi_loss_vec.push_back(pi_loss_old_val);
    
    for(int64_t i = 0; i < opt.train_pi_iter(); i++)
    {
        optimizer_pi->zero_grad();
        torch::Tensor pi_loss;
        PPO::ExtraInfo info;
        std::tie(pi_loss, info) = compute_pi_loss(batch);
        // cout << "pi_loss: " << pi_loss.item<float>() << endl;
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
        // cout << "v_loss: " << v_loss.item<float>() << endl;
        v_loss.backward();
        optimizer_q->step();
    }    

    this->_update_count++;
}

}; // namespace PPO