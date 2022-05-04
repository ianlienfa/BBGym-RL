#include "oneRjSumCjSearch.h"

OneRjSumCjSearch::OneRjSumCjSearch(){
    #if LABELER == labeler_bynet
        std::cout << "Required to label by net, however net is not initialized, exiting" << std::endl;
        exit(LOGIC_ERROR);
    #endif
}

void OneRjSumCjSearch::fill_graph(OneRjSumCjGraph *graph) {
    this->graph = graph;
}

bool OneRjSumCjSearch::get_find_optimal() {
    return this->graph->contours.empty();
}

/* initializes graph for OneRjSumCj */
OneRjSumCjGraph OneRjSumCjSearch::init(OneRjSumCjNode rootProblem) {
    
    if(rootProblem.processing_time.empty() || rootProblem.release_time.empty() || rootProblem.job_weight.empty()){
        cerr << "Error: The root problem is not valid." << endl;
        exit(INVALID_INPUT);
    }

    // copy by value, could be improved
    OneRjSumCjGraph graph(rootProblem.processing_time, rootProblem.release_time, rootProblem.job_weight, rootProblem.jobs_num);

/* initialization for CBFS data structures */
#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
    PriorityQueue<OneRjSumCjNode> contour_0(OneRjSumCjNode::cmpr);
    contour_0.push(OneRjSumCjNode(B(0), Vi(), 0));
    graph.contours.insert(make_pair(0, contour_0));
    graph.current_contour_iter = graph.contours.begin();
#endif

/* 
since we need to check for the equalness of start time of all unordered jobs, 
we can save them in a ordered map
*/
#if (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
    graph.contours.resize(OneRjSumCjNode::jobs_num+1);
    graph.current_level = 0;
    graph.contours[0].push_back(OneRjSumCjNode(B(0), Vi(), 0));
#endif

    return graph;
}

#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
OneRjSumCjNode OneRjSumCjSearch::search_next() {
    // pop element from the current contour
    #if DEBUG_LEVEL >=2         
        cout << "current contour: " << graph->current_contour_iter->first << endl;
        cout << "current contour size: " << graph->current_contour_iter->second.size() << endl;
        cout << endl;
    #endif
    OneRjSumCjNode current_element = this->graph->current_contour_iter->second.top();
    this->graph->current_contour_iter->second.extract();
    return current_element;        
}
#elif (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
OneRjSumCjNode OneRjSumCjSearch::search_next() {
    // pop element from the current contour    
    OneRjSumCjNode current_element = this->graph->contours[this->graph->current_level].front();
    // this->graph->contours[this->graph->current_level].pop_front();
    return current_element;
}
#endif

#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
vector<OneRjSumCjNode> OneRjSumCjSearch::update_graph(OneRjSumCjNode current_node, vector<OneRjSumCjNode> branched_nodes) {
    (void) current_node;
    // update node count
    this->graph->searched_node_num += branched_nodes.size();
    #if DEBUG_LEVEL >=2        
    cout << "searched_node_num: " << this->graph->searched_node_num << endl;
    #endif
    // push the branched nodes into the contour
    for(vector<OneRjSumCjNode>::iterator it = branched_nodes.begin(); it != branched_nodes.end(); ++it){
        
        // label and push
        StateInput stateInput(current_node, *it, *this->graph);
        vector<float> s = stateInput.get_state_encoding();  
        vector<float> contour_snap = this->graph->get_contour_snapshot(labeler->max_num_contour);      
        bool inference = INF_MODE;
        float label = 0;   
        float prob, noise;
        vector<float> softmax;
        ActorOut out;
        if(!inference)
        {     
            if(labeler->epoch < labeler->update_start_epoch)
            {                
                out = (*labeler).train(s, contour_snap, DDPRLabeler::OperatorOptions::RANDOM);
                std::tie(out, label) = (*labeler).concept_label_decision(out); // plain interpretation  
                assertm("label_decision(): label is out of range", (label > 0) && (label < labeler->action_range.second));
                #if TORCH_DEBUG == 1                        
                if(std::isnan(label))
                    throw std::runtime_error("Labeler returned NaN");
                #endif                
            }
            else if(labeler->epoch < labeler->num_epoch)
            {
                // for tensor related data, pass by value would be better
                out = (*labeler).train(s, contour_snap, DDPRLabeler::OperatorOptions::TRAIN);
                std::tie(out, label) = (*labeler).concept_label_decision(out, true); // exploration interpretation                
                assertm("label_decision(): label is out of range", (label > 0) && (label < labeler->action_range.second));
                #if TORCH_DEBUG == 1                        
                if(std::isnan(label))
                    throw std::runtime_error("Labeler returned NaN");
                #endif
            }
            else
            {
                // last epoch do one inference
                cout << "INFERENCING ... " << endl;
                out = (*labeler).train(s, contour_snap, DDPRLabeler::OperatorOptions::INFERENCE);
                std::tie(out, label) = (*labeler).concept_label_decision(out); // plain interpretation  
                assertm("label_decision(): label is out of range", (label > 0) && (label < labeler->action_range.second));
            }
            
            std::tie(prob, noise, softmax) = out; // copy for buffer use
            assertm("prob not in range", 0.0 <= prob && prob <= 1.0);
            assertm("noise not in range", -1.0 <= noise && noise <= 1.0);
            #ifndef NDEBUG
            for(auto & p : softmax)
                assertm("softmax not in range", 0.0 <= p && p <= 1.0);
            #endif
            
            vector<float> action_prep = {prob, noise};
            action_prep.insert(action_prep.end(), softmax.begin(), softmax.end());
            cout << "action_prep: " << endl << action_prep << endl;

            if(!labeler->buffer->isin_prep()) 
            {     
                labeler->buffer->enter_data_prep_section();
                labeler->buffer->s_prep = s;
                labeler->buffer->a_prep = action_prep;
                labeler->buffer->contour_snapshot_prep = contour_snap;
            }
            else
            {
                // Finish the last data prep section
                labeler->buffer->s_next_prep = s;   
                labeler->buffer->contour_snapshot_next_prep = contour_snap;
                // If the new node is search instead of branching nodes from same parent, reward = -1
                labeler->buffer->reward_prep = (it == branched_nodes.begin()) ? node_reward : neg_zero_reward;
                labeler->buffer->done_prep = 0.0;
                labeler->buffer->leave_data_prep_section();
                labeler->buffer->submit();

                // track reward
                this->graph->accu_reward += labeler->buffer->reward_prep;
                // cout << "current reward: " << labeler->buffer->reward_prep << endl;
                
                // start the new data prep section
                labeler->buffer->enter_data_prep_section();
                labeler->buffer->s_prep = s;
                labeler->buffer->a_prep = action_prep;
                labeler->buffer->contour_snapshot_prep = contour_snap;
            }
        }
        else
        {
            label = (*labeler)(s, contour_snap, DDPRLabeler::OperatorOptions::INFERENCE);  
        }        
        // increase the step count
        labeler->step++;        

        #ifndef NDEBUG
        auto current_snap = this->graph->get_contour_snapshot(labeler->max_num_contour);
        for(size_t i = 0; i < contour_snap.size(); i++)
        {
            assertm("contour snap should have same value as current snap", current_snap[i] == contour_snap[i]);
        }
        #endif

        // Push the label(action) into contour : step()
        assertm("label_decision(): label is out of range", (label > 0) && (label < 5));
        this->graph->clip_insert(labeler->max_num_contour, this->graph->contours, *it, label);
    }

    // locate the next contour
    map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>::iterator next_iter = this->graph->current_contour_iter;
    // if there's contour left after deletion, cycle back or go to the next
    if(this->graph->contours.size() > 1)
    {        
        next_iter++;
        if(next_iter == this->graph->contours.end())
        {
            next_iter = this->graph->contours.begin();
        }        
    }

    // clean current contour if needed
    if(this->graph->current_contour_iter->second.empty())
    {
        // if the current contour is empty, we need to find the next contour
        map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>::iterator current_iter = this->graph->current_contour_iter;       
 
        if(this->graph->contours.size() <= 1) /* early leaving from the environment */
        {
            // only one contour left and the contour is empty, we need to stop
            this->graph->optimal_found = true;
            
            /* complete the incomplete data prep section */
            // create a dummy stateInput only to call the labeler for a terminal state representation return            
            StateInput dummy(current_node, current_node, *this->graph);
            labeler->buffer->s_next_prep = dummy.get_state_encoding();            
            labeler->buffer->contour_snapshot_next_prep = this->graph->get_contour_snapshot(labeler->max_num_contour);        
            labeler->buffer->reward_prep = pos_zero_reward;
            labeler->buffer->done_prep = 1.0;
            labeler->buffer->leave_data_prep_section();
            labeler->buffer->submit();

            // track reward            
            this->graph->accu_reward += labeler->buffer->reward_prep;
            // cout << "current reward: " << labeler->buffer->reward_prep << endl;
        }
        this->graph->contours.erase(current_iter);
    }
    
    // update the current contour
    this->graph->current_contour_iter = next_iter;
    
    #if DEBUG_LEVEL >= 0
    if(labeler->step % 1 == 0)
    {
        cout << "labeler.step: " << labeler->step << endl;
        cout << "------------------" << endl;
        for(auto it = this->graph->contours.begin(); it != this->graph->contours.end(); ++it)
        {
            std::cout << std::setprecision(6) << " " << it->first << " " << it->second.size() << "\n";
        }
        cout << "current_iter: " << this->graph->current_contour_iter->first << endl;
    }
    #endif
    if(this->graph->avg_reward){ 
        this->graph->avg_reward = this->graph->avg_reward * 0.9 + this->graph->accu_reward * 0.1;
    }
    else{
        this->graph->avg_reward = this->graph->accu_reward;
    }
    return branched_nodes;
}
#elif (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
vector<OneRjSumCjNode> OneRjSumCjSearch::update_graph(OneRjSumCjNode current_node, vector<OneRjSumCjNode> branched_nodes) {

    #if (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
    this->graph->contours[this->graph->current_level].pop_front();
    if(branched_nodes.size() == 0)
    {    
        if(this->graph->current_level == 0)
        {
            cout << "optimal solution found" << endl;
            for(auto it: this->graph->min_seq)
                cout << it << " ";
                cout << endl;
            cout << "optimal solution WjCj: " << this->graph->min_obj << endl;  
            this->graph->optimal_found = true;
        }
    }       
    #endif
    (void) current_node;
    this->graph->searched_node_num++;
    this->graph->current_level++;
    // push branched nodes into the contour
    for(vector<OneRjSumCjNode>::iterator it = branched_nodes.begin(); it != branched_nodes.end(); ++it){
        this->graph->contours[this->graph->current_level].push_back(*it);
    }
    #if DEBUG_LEVEL >= 2
    cout << "last level: " << this->graph->current_level - 1 << "[" << endl;    
    for(auto it: this->graph->contours[this->graph->current_level - 1]){
        cout << it << endl;
    }
    cout << endl << "]" << endl;
    cout << "current level: " << this->graph->current_level << "[" << endl;
        for(auto it: this->graph->contours[this->graph->current_level]){
        cout << it << endl;
    }
    cout << endl << "]" << endl;
    cout << "-" << endl;
    cout << "searched_node_num: " << this->graph->searched_node_num << endl;    
    cout << "current_level: " << this->graph->current_level << endl;
    #endif
    // update current livel if the current level nothing is needed to be search for    
    while(this->graph->contours[this->graph->current_level].size() == 0 && this->graph->current_level != 0)
    {
        #if DEBUG_LEVEL >= 1
        cout << this->graph->current_level << " is empty" << endl;
        #endif
        this->graph->current_level--;
        // this->graph->contours[this->graph->current_level].pop_front();
    }
    if(this->graph->current_level == 0)
    {
        #if DEBUG_LEVEL >= 1
        cout << "optimal solution found" << endl;
        for(auto it: this->graph->min_seq)
            cout << it << " ";
            cout << endl;
        cout << "optimal solution WjCj: " << this->graph->min_obj << endl;  
        #endif
        this->graph->optimal_found = true;
    }
    return branched_nodes;
}
#endif


#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
bool OneRjSumCjSearch::is_incumbent(const OneRjSumCjNode &current_node) {
    return ((((int)current_node.seq.size()) == OneRjSumCjNode::jobs_num));
}
#elif (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
bool OneRjSumCjSearch::is_incumbent(const OneRjSumCjNode &current_node) {
    (void) current_node;
    return false;
}
#endif

void OneRjSumCjSearch::history_fill()
{
    
}