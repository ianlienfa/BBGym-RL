#include <iostream>
#include "user_def/oneRjSumCj_engine.h"
#include <unistd.h>

OneRjSumCj_engine::OneRjSumCj_engine()
{
    this->rootProblem = OneRjSumCjNode();
}

OneRjSumCj_engine::OneRjSumCj_engine(OneRjSumCjGraph graph, OneRjSumCjSearch searcher, OneRjSumCjBranch brancher, OneRjSumCjPrune pruner, LowerBound lowerbound)
: graph(graph), searcher(searcher), brancher(brancher), pruner(pruner), lowerbound(lowerbound)
{
    this->rootProblem = OneRjSumCjNode();
}

OneRjSumCjGraph OneRjSumCj_engine::solve(OneRjSumCjNode rootProblem)
{    
    PRE_SOLVE_PRINT_CONFIG();

    // the root graph
    this->graph = searcher.init(rootProblem);

    // save the reference of graph in all modules
    searcher.fill_graph(&graph);
    brancher.fill_graph(&graph);
    pruner.fill_graph(&graph);

    // a incumbent init for Lu & Sal
    OneRjSumCjNode initIncumbent = OneRjSumCjNode::getInitESTSeq();
    this->graph.min_obj = initIncumbent.weighted_completion_time;
    this->graph.min_seq = initIncumbent.seq;

    #if DEBUG_LEVEL >= 3
    cout << "init incumbent: " << initIncumbent.weighted_completion_time << ", { ";
    for(auto it : initIncumbent.seq)
        cout << it << " ";
        cout << "}" << endl;
    #endif
    
    // start the branch and bound algorithm
    while(!graph.optimal_found){
        // searcher search on to a new node, make changes on its internal data structure, then return the new node
        this->graph.current_node = searcher.search_next();

        #if DEBUG_LEVEL >= 2
                cout << "==============================New Search==============================" << endl;
                std::cout << this->graph.current_node << std::endl;
                cout << "current incumbent value: " << this->graph.min_obj << endl;
        #endif
        
        // the solver engine knows about the problem, update the current best node if feasible
        if(searcher.is_incumbent(this->graph.current_node))
        {
            this->update_incumbent(this->graph.current_node);
        }

        // brancher evaluate the node and decide whether to branch, and return the reference branched nodes (in vector)
        vector<OneRjSumCjNode> branched_nodes = brancher.branch(this->graph.current_node, lowerbound);

        // pruner evaluate the branched nodes and decide remove the nodes to prune
        pruner.prune(branched_nodes);
        

        // update the graph
        searcher.update_graph(this->graph.current_node, branched_nodes);
    }

    POST_SOLVE_PRINT_CONFIG(graph);
    return graph;
}

/*--------------------------------------------------------------------------
 * modified: this->graph.min_obj & this->graph.min_seq
--------------------------------------------------------------------------*/
void OneRjSumCj_engine::update_incumbent(const OneRjSumCjNode &current_node) {
    double obj = this->objSolve(current_node);   
    if(obj < this->graph.min_obj)
    {
    #if DEBUG_LEVEL >= 1
        std::cout << "========update_incumbent=======" << std::endl;
        std::cout << "obj: " << this->graph.min_obj << std::endl;
    #endif
        this->graph.min_obj = obj;
        this->graph.min_seq = current_node.seq;
    }
}


TIME_TYPE OneRjSumCj_engine::objSolve(const OneRjSumCjNode &current_node) {
        // feasibility check
    if(current_node.processing_time.size() != current_node.release_time.size() ||
        current_node.job_weight.empty())
    {
        cout << "[main.cpp::objSolve] INVALID_INPUT: the input data is not in the correct format" << endl;
        exit(INVALID_INPUT);
    }
    if(current_node.seq.size() != current_node.jobs_num)
    {
        cout << "[main.cpp::objSolve] LOGIC_ERROR: the sequence is not completed!" << endl;
        exit(LOGIC_ERROR);
    }

    // calculate the objective value
    auto start_time = [](int current_time, int release_date){return std::max(current_time, release_date);};
    int current_time = 0;
    int obj = 0;
    vector<int> start_arr;
    vector<int> complete_arr;
    for(size_t i = 0; i < current_node.seq.size(); i++)
    {
        int s_i = start_time(current_time, current_node.release_time[current_node.seq[i]]);
        int c_i = s_i + current_node.processing_time[current_node.seq[i]];
        current_time = c_i;
        obj += c_i;
    }
    return obj;

}

void print_config(){
    cout << "=============== Search Started ===============" << endl;
    cout << "Working on instance: " << OneRjSumCjNode::instance_name << endl;
    cout << "Instance size = " << OneRjSumCjNode::jobs_num << endl;

    #if DEBUG_LEVEL >= 1
    cout << "configuration:" << endl;
    cout << "===============" << endl;

    for(auto it = OneRjSumCjNode::processing_time.begin(); it != OneRjSumCjNode::processing_time.end(); it++)
        cout << "processing_time[" << it - OneRjSumCjNode::processing_time.begin() << "] = " << *it << endl;

    for(auto it = OneRjSumCjNode::release_time.begin(); it != OneRjSumCjNode::release_time.end(); it++)
        cout << "release_time[" << it - OneRjSumCjNode::release_time.begin() << "] = " << *it << endl;

    for(auto it = OneRjSumCjNode::job_weight.begin(); it != OneRjSumCjNode::job_weight.end(); it++)
    cout << "job_weight[" << it - OneRjSumCjNode::job_weight.begin() << "] = " << *it << endl;        

    cout << "jobs_num = " << OneRjSumCjNode::jobs_num << endl;
    cout << "bit mask = " << OneRjSumCjNode::jobs_mask  << endl;
    sleep(2);
    #endif
}

void post_print_config(const OneRjSumCjGraph &graph)
{
    cout << endl;
    cout << "--------------- Search Result ---------------" << endl;
    cout << "number of searched nodes: " << graph.searched_node_num << endl;
    cout << "min_obj = " << graph.min_obj << endl;
    cout << "min_seq = ";
    for(auto it = graph.min_seq.begin(); it != graph.min_seq.end(); it++)
        cout << *it << " ";
    cout << endl;
    cout << "=============== Search Ended ===============" << endl;
    cout << endl << endl;
}