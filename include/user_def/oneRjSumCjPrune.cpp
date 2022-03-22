#include "oneRjSumCjPrune.h"

/* Definition for static members */
vector<prune_func_pt> OneRjSumCjPrune::prune_funcs;
vector<prune_func_safept> OneRjSumCjPrune::safe_prune_funcs;


/*----------------------------------------------------------------------------
 * function: prune method -- bianco1982_1
 * Specification: dominace rule: smaller pi/wi, smaller start time
 * Usage: 
 * Constructor: 
------------------------------------------------------------------------------
----------------------------------------------------------------------------*/
void OneRjSumCjPrune::fill_graph(OneRjSumCjGraph *graph) {
    this->graph = graph;
}

void OneRjSumCjPrune::prune_func_init(vector<prune_func_pt>& prune_funcs) {
    this->prune_funcs = prune_funcs;
}

void OneRjSumCjPrune::prune_func_init() {
    vector<prune_func_pt> prune_funcs = {
    };
    this->prune_funcs = prune_funcs;
}

/*----------------------------------------------------------------------------
 * function: prune method -- basic prune
 * Specification: remove nodes with lowerbound >= upperbound
 * Usage: Problem specific pruning function can be defined here, 
 *        and can be chosen in main.cpp to inspect the pruning effect
 * Constructor:
------------------------------------------------------------------------------*/
void OneRjSumCjPrune::prune(vector<OneRjSumCjNode> &branched_nodes){

    #if DEBUG_LEVEL >= 1
    cout << ">> Pruning starts >>" << endl;
    #endif

    for(vector<prune_func_pt>::iterator it = prune_funcs.begin(); it != prune_funcs.end(); it++)
    {
        (*it)(branched_nodes, *graph);
    }
    for(vector<prune_func_safept>::iterator it = safe_prune_funcs.begin(); it != safe_prune_funcs.end(); it++)
    {
        (*it)(branched_nodes, *graph);
    }

    #if DEBUG_LEVEL >= 1
    cout << "<< Pruning ends <<" << endl;
    #endif

}

/* Pruning function can be added here... 
   The naming convention is prune__{PROBLEM_NAME}__{PRUNING_STRATEGY}__{PRUNING_FUNCTION_NAME} 
*/
void pruneIncumbentCmpr(vector<OneRjSumCjNode>& branched_nodes, const OneRjSumCjGraph &graph) {
    for(vector<OneRjSumCjNode>::iterator it = branched_nodes.begin(); it != branched_nodes.end(); it++)
    {
        if(it->lb >= graph.min_obj)
        {
            #if DEBUG_LEVEL == 2
            cout << "pruneIncumbentCmpr: remove node " << *it << endl;
            #endif     
            branched_nodes.erase(it);
            it--;
        }
    }
}

void prune__OneRjSumCj__LU_AND_SAL__Theorem1(vector<OneRjSumCjNode>& branched_nodes, OneRjSumCjGraph &graph) {    
    if(branched_nodes.size() == 0)
        return;
    #if (VALIDATION_LEVEL == validation_level_HIGH)
        for(size_t i = 0; i < branched_nodes.size()-1; i++)
        {
            if(branched_nodes[i].earliest_start_time > branched_nodes[i+1].earliest_start_time)
            {
                cout << "Error:[oneRjSumCjPrune.cpp:prune__OneRjSumCj__LU_AND_SAL__Theorem1]: V_j is not sorted by earliest start time!" << endl;
                exit(LOGIC_ERROR);
            }
        }
    #endif
    /* if all earliest start time are the same (implying that every job is released) 
       then update incumbent solution with WSPT rule to find the corresponding feasible solution for the current partial sequence,
       prune all nodes, and go back       
    */
   if(branched_nodes[0].earliest_start_time == branched_nodes[branched_nodes.size()-1].earliest_start_time && int(branched_nodes.size()) == graph.current_node.get_unfinished_jobs_num())
   {
        // compute WSPT seq (id, p/w)
        vector<int> V_j;
        vector<pair<int, int>> seq_wspt;
        for(size_t i = 0; i < branched_nodes.size(); i++)
        {
            V_j.push_back(branched_nodes[i].seq.back());
        }
        for(size_t i = 0; i < V_j.size(); i++)
        {
            seq_wspt.push_back(make_pair(V_j[i], OneRjSumCjNode::processing_time[V_j[i]]/OneRjSumCjNode::job_weight[V_j[i]]));
        }
        sort(seq_wspt.begin(), seq_wspt.end(), [](const pair<int, int> &a, const pair<int, int> &b) {return a.second < b.second;});

        // update incumbent solution
        OneRjSumCjNode incumbent(branched_nodes[0]);
        incumbent.seq.pop_back(); // clear branched node
        
        // push new partial sequence to the back
        for(size_t i = 0; i < seq_wspt.size(); i++)
        {
            incumbent.seq.push_back(seq_wspt[i].first);
            incumbent.is_processed.set(seq_wspt[i].first);
        }
        pair<int, int> wjcj_cj = OneRjSumCjNode::getObj(incumbent.seq);
        incumbent.weighted_completion_time = wjcj_cj.first;
        incumbent.completion_time = wjcj_cj.second;
        if(incumbent.weighted_completion_time < graph.min_obj)
        {
            graph.min_obj = incumbent.weighted_completion_time;
            graph.min_seq = incumbent.seq;
            #if DEBUG_LEVEL >= 1
            cout << "new incumbent: " << incumbent << endl;        
            #endif
        }
        // clear branched_nodes
        branched_nodes.clear();
        #if DEBUG_LEVEL == 2
        cout << "prune__OneRjSumCj__LU_AND_SAL__Theorem1 triggered, multiple node pruned" << endl;
        #endif
   }
}
// void prune__OneRjSumCj__LU_AND_SAL__basicFilter(vector<OneRjSumCjNode>& branched_nodes, const OneRjSumCjGraph &graph) {    
    
// }
<<<<<<< HEAD
// #endif
=======
>>>>>>> template/main
