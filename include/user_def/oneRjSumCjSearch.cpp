#include "oneRjSumCjSearch.h"

OneRjSumCjSearch::OneRjSumCjSearch(){
    #if LABELER == labeler_bynet
        std::cout << "Required to label by net, however net is not initialized, exiting" << std::endl;
        exit(LOGIC_ERROR);
    #endif
}

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
        float label = (*labeler)(s);  

        if(!labeler->buffer->isin_prep()) 
        {     
            labeler->buffer->enter_data_prep_section();
            labeler->buffer->s_prep = s;
            labeler->buffer->label_prep = label;
        }
        else
        {
            // Finish the last data prep section
            labeler->buffer->s_next_prep = s;            
            // If the new node is search instead of branching nodes from same parent, reward = -1
            labeler->buffer->reward_prep = (it == branched_nodes.begin()) ? -1.0 : 0.0;
            labeler->buffer->done_prep = 0.0;
            labeler->buffer->leave_data_prep_section();
            labeler->buffer->submit();
            
            // start the new data prep section
            labeler->buffer->enter_data_prep_section();
            labeler->buffer->s_prep = s;
            labeler->buffer->label_prep = label;
        }

        map<int, PriorityQueue<OneRjSumCjNode>>::iterator target_contour_iter = this->graph->contours.find(label);
        if(target_contour_iter == this->graph->contours.end())
        {
            PriorityQueue<OneRjSumCjNode> pq_insert(OneRjSumCjNode::cmpr);
            pq_insert.push(*it);
            #if DEBUG_LEVEL >=2
                cout << "inserting node to contour " << label << ": " << *it  << endl;
            #endif
            this->graph->contours.insert(make_pair(label, pq_insert));
        }
        else
        {
            target_contour_iter->second.push(*it);
            #if DEBUG_LEVEL >=2
                cout << "inserting node to contour " << label << ": " << *it  << endl;
            #endif
        }        
    }

    // clean contour
    if(this->graph->current_contour_iter->second.empty())
    {
        // if the current contour is empty, we need to find the next contour
        map<int, PriorityQueue<OneRjSumCjNode>>::iterator current_iter = this->graph->current_contour_iter;
        
        // if there's contour left after deletion, cycle back or go to the next
        if(this->graph->contours.size() > 1)
        {
            if(this->graph->current_contour_iter != this->graph->contours.end())
            {
                ++this->graph->current_contour_iter;
            }
            else
            {
                this->graph->current_contour_iter = this->graph->contours.begin();
            }
        }
        else
        {
            // only one contour left and the contour is empty, we need to stop
            this->graph->optimal_found = true;

            /* complete the incomplete data prep section */
            // create a dummy stateInput only to call the labeler for a terminal state representation return
            StateInput dummy(current_node, current_node, *this->graph);
            labeler->buffer->s_next_prep = dummy.get_state_encoding(true);            
            labeler->buffer->reward_prep = 0.0;
            labeler->buffer->done_prep = 1.0;
            labeler->buffer->leave_data_prep_section();
            labeler->buffer->submit();
        }
        this->graph->contours.erase(current_iter);
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