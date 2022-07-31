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
    #if DEBUG_LEVEL == 2
    cout << "prune__OneRjSumCj__LU_AND_SAL__Theorem1: enter "<< endl;
    #endif     

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
        // reorder the branch nodes by WSPT
        vector<pair<OneRjSumCjNode*, float>> wspt_node;
        for(size_t i = 0; i < branched_nodes.size(); i++)
        {
            int last_job = branched_nodes[i].seq.back();
            wspt_node.push_back(make_pair(&branched_nodes[i], OneRjSumCjNode::processing_time[last_job]/OneRjSumCjNode::job_weight[last_job]));
        }
        sort(wspt_node.begin(), wspt_node.end(), [](const pair<OneRjSumCjNode*, float>& a, const pair<OneRjSumCjNode*, float>& b) {
            return (a.second == b.second) ? (a.first->seq.back() < b.first->seq.back()) : (a.second < b.second);
        });

        // extend the first node to be the incumbent solution
        OneRjSumCjNode incumbent_candidate(*wspt_node[0].first);
        OneRjSumCjNode node_to_branch(*wspt_node[0].first);      

        // fill the remaining jobs to the incumbent
        for(size_t i = 1; i < wspt_node.size(); i++)
        {
            incumbent_candidate.seq.push_back(wspt_node[i].first->seq.back());
            incumbent_candidate.is_processed.set(wspt_node[i].first->seq.back());
        }                
        pair<int, int> wjcj_cj = OneRjSumCjNode::getObj(incumbent_candidate.seq);
        incumbent_candidate.weighted_completion_time = wjcj_cj.first;
        incumbent_candidate.completion_time = wjcj_cj.second;

        if(incumbent_candidate.weighted_completion_time < graph.min_obj)
        {
            graph.min_obj = incumbent_candidate.weighted_completion_time;
            graph.min_seq = incumbent_candidate.seq;
            #if DEBUG_LEVEL >= 1
            cout << "new incumbent: " << incumbent_candidate << endl;        
            #endif
        }

        // clear branched_nodes     
        branched_nodes.clear();
        branched_nodes.push_back(node_to_branch);
        #if DEBUG_LEVEL == 2
        cout << "prune__OneRjSumCj__LU_AND_SAL__Theorem1 triggered, multiple node pruned" << endl;
        #endif
    }
}
