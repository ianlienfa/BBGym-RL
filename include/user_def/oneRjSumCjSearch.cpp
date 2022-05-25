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
    using namespace PPO;

    (void) current_node;
    // update node count
    this->graph->searched_node_num += branched_nodes.size();

    // push the branched nodes into the contour
    for(vector<OneRjSumCjNode>::iterator it = branched_nodes.begin(); it != branched_nodes.end(); ++it){
                
        MEASURE(stateInput_measurer, "stateInput",
        StateInput stateInput(current_node, *it, *this->graph);
        );
        MEASURE(get_state_encoding_measurer, "get_state_encoding",
        vector<float> s = stateInput.get_state_encoding();  
        );
        torch::Tensor tensor_s = torch::from_blob(s.data(), {1, int64_t(s.size())}).clone();        

        // check if trajectory has finished
        if(labeler->step() == labeler->opt.steps_per_epoch() - 1)
        {
            // compute value of current state and call finish path
            const auto &stepout = labeler->net->step(tensor_s);
            labeler->buffer->finish_epoch(stepout.v);
        }

        // label and push
        float label = (*labeler)(s);

        // Push the label(action) into contour : step()
        assertm("label_decision(): label is out of range", (label > 0) && (label < 5));
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
            // update buffer (s, a, '', v, logp) -> (s, a, r, v, logp)
            labeler->buffer->prep.r() = pos_zero_reward;
            labeler->buffer->submit();

            // track reward           
            const auto &prep_reader = labeler->buffer->prep;
            this->graph->accu_reward += prep_reader.r();
            // cout << "current reward: " << labeler->buffer->reward_prep << endl;
        }
        this->graph->contours.erase(current_iter);
    }
    
    // update the current contour
    this->graph->current_contour_iter = next_iter;
    
    #if DEBUG_LEVEL >= 0
    if(labeler->step() % 1 == 0)
    {
        cout << "labeler.step: " << labeler->step() << endl;
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

    std::ofstream outfile;
    outfile.open("../saved_model/rewards.txt", std::ios_base::app);  
    outfile << this->graph->avg_reward << ", ";  
    outfile.close();

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