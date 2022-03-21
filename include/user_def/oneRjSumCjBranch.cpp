#include "user_def/oneRjSumCjBranch.h"
#include "util/LowerBound.h"

/*----------------------------------------------------------------------------
 Branch strategy for OneRjSumCj:
  Branch all unvisited jobs 
------------------------------------------------------------------------------*/
#if(BRANCH_STRATEGY == branchOneRjSumCj_PLAIN)
vector<OneRjSumCjNode> OneRjSumCjBranch::branch(const OneRjSumCjNode &current_node, LowerBound &lower_bound) {
    vector<OneRjSumCjNode> branched_nodes;
    for(size_t i = 1; i <= current_node.jobs_num; i++)
    {
        if(current_node.is_processed[i]) continue;
        OneRjSumCjNode new_node(current_node);
        new_node.is_processed.set(i);
        new_node.seq.push_back(i);        
        new_node.lb = lower_bound(new_node);
        branched_nodes.push_back(new_node);
    }
    return branched_nodes;
}
#elif (BRANCH_STRATEGY == branchOneRjSumCj_LU_AND_SAL)
OneRjSumCjNode branch_job(const OneRjSumCjNode &current_node, size_t job_id, LowerBound &lower_bound) {
    OneRjSumCjNode new_node(current_node);
    new_node.is_processed.set(job_id);
    new_node.seq.push_back(job_id);
    new_node.earliest_start_time = std::max(OneRjSumCjNode::release_time[job_id], current_node.completion_time);
    new_node.completion_time = new_node.earliest_start_time + OneRjSumCjNode::processing_time[job_id]; 
    new_node.weighted_completion_time = new_node.completion_time * OneRjSumCjNode::job_weight[job_id] + current_node.weighted_completion_time;;   
    new_node.lb = lower_bound(new_node);
    return new_node;
}
vector<OneRjSumCjNode> OneRjSumCjBranch::branch(const OneRjSumCjNode &current_node, LowerBound &lower_bound) {
    #if DEBUG_LEVEL >= 1
    cout << ">> Branching starts >>" << endl;
    #endif
    // Create Dense Set of Jobs
    vector<OneRjSumCjNode> V_j;
    int min_Cj = INT32_MAX;
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++)
    {
        if(current_node.is_processed[i]) continue;
        OneRjSumCjNode new_node = branch_job(current_node, i, lower_bound);        
        if(new_node.completion_time < min_Cj) 
            min_Cj = new_node.completion_time;
        V_j.push_back(new_node);        
    }
    #if (DEBUG_LEVEL >= 3)
        cout << "Branching..." << endl;
        cout << "V_j: [" << endl;
        for(auto it : V_j)
            cout << it << endl;
        cout << "]" << endl;
        cout << "min_Cj: " << min_Cj << endl;
    #endif
    sort(V_j.begin(), V_j.end(), OneRjSumCjNode::cmpr);
    for(int i = V_j.size()-1; i >= 0; i--)
    {
        if(V_j[i].earliest_start_time >= min_Cj)
        {
            V_j.pop_back();
            #if (DEBUG_LEVEL >= 2)
            cout << "PrePruned job " << V_j[i] << " by collary 1" << endl;
            #endif
        }
    }
  
    #if DEBUG_LEVEL >= 2
    cout << "current_node:" << graph->current_node << endl;
    cout << "branched node: [" << endl;
    for(auto it: V_j)        
        cout << it << endl;            
    cout << "]" << endl;
    #endif
    #if DEBUG_LEVEL >= 1
    cout << "<< Branching ends <<" << endl;
    #endif


    return V_j;
}
#endif
void OneRjSumCjBranch::fill_graph(OneRjSumCjGraph *graph) {
    this->graph = graph;
}