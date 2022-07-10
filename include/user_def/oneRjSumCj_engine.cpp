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
    MEASURE(fill_graph_measurer, "fill_graph_measurer",
    /// ---- measure target ---------------------------------------
    searcher.fill_graph(&graph);
    brancher.fill_graph(&graph);
    pruner.fill_graph(&graph);
    /// -----------------------------------------------------------
    );

    // a incumbent init for Lu & Sal
    OneRjSumCjNode initIncumbent = OneRjSumCjNode::getInitESTSeq();
    this->graph.min_obj = initIncumbent.weighted_completion_time;
    this->graph.min_seq = initIncumbent.seq;

    #if DEBUG_LEVEL >= 2
    cout << "init incumbent: " << initIncumbent.weighted_completion_time << ", { ";
    for(auto it : initIncumbent.seq)
        cout << it << " ";
        cout << "}" << endl;
    #endif
    
    // start the branch and bound algorithm
    while(!graph.optimal_found){
        SOLVE_CALLBACK(this);

        // searcher search on to a new node, make changes on its internal data structure, then return the new node
        MEASURE(search_next_measurer, "search_next",
        this->graph.current_node = searcher.search_next();
        );
        #if DEBUG_LEVEL >= 2
            cout << "==============================New Search==============================" << endl;
            std::cout << this->graph.current_node << std::endl;
            cout << "current incumbent value: " << this->graph.min_obj << endl;                
            cout << "current incumbent :";
            for(auto it : this->graph.min_seq)
                cout << it << " ";
            cout << endl;
        #endif
        
        // the solver engine knows about the problem, update the current best node if feasible
        if(searcher.is_incumbent(this->graph.current_node))
        {
            this->update_incumbent(this->graph.current_node);
        }

        // brancher evaluate the node and decide whether to branch, and return the reference branched nodes (in vector)
        MEASURE(branching_measurer, "branching",
        vector<OneRjSumCjNode> branched_nodes = brancher.branch(this->graph.current_node, lowerbound);
        );

        // pruner evaluate the branched nodes and decide remove the nodes to prune
        MEASURE(pruning_measurer, "pruning",
        pruner.prune(branched_nodes);
        );

        // update the graph
        MEASURE(update_graph_measurer, "update_graph",
        searcher.update_graph(this->graph.current_node, branched_nodes);
        );
    }

    OPTIMAL_FOUND_CALLBACK(this);
    POST_SOLVE_PRINT_CONFIG(graph);
    return graph;
}

/*--------------------------------------------------------------------------
 * modified: this->graph.min_obj & this->graph.min_seq
--------------------------------------------------------------------------*/
void OneRjSumCj_engine::update_incumbent(const OneRjSumCjNode &current_node) {
    assertm("weighted completion time of current_node is not computed", current_node.weighted_completion_time != 0.0);
    double obj = current_node.weighted_completion_time;
    if(obj < this->graph.min_obj)
    {
    #if DEBUG_LEVEL >= 1
        std::cout << "========update_incumbent=======" << std::endl;
        std::cout << "current node: " << current_node << std::endl;
        std::cout << "old obj: " << this->graph.min_obj << ", new obj: " << obj << std::endl;
    #endif
        this->graph.min_obj = obj;
        this->graph.min_seq = current_node.seq;
    }
}


TIME_TYPE OneRjSumCj_engine::objSolve(const OneRjSumCjNode &current_node) {
    // feasibility check
    assertm("[main.cpp::objSolve] INVALID_INPUT: the input data is not in the correct format", current_node.processing_time.size() == current_node.release_time.size());
    assertm("[main.cpp::objSolve] INVALID_INPUT: job_weight should not be empty", (!current_node.job_weight.empty()));
    assertm("[main.cpp::objSolve] LOGIC_ERROR: the sequence is not completed!", current_node.seq.size() == current_node.jobs_num);

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
        obj += c_i * current_node.job_weight[current_node.seq[i]];
    }
    return obj;
}

void print_config(){
    #if PURE_SEARCH_NODE_NUM != 1
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
    #endif
    #endif
}

void post_print_config(const OneRjSumCjGraph &graph)
{
    #if PURE_SEARCH_NODE_NUM != 1
    cout << endl;
    cout << "--------------- Search Result ---------------" << endl;
    cout << "instance: " << OneRjSumCjNode::instance_name << endl;
    cout << "number of searched nodes: " << graph.searched_node_num << endl;
    cout << "min_obj = " << graph.min_obj << endl;
    cout << "min_seq = ";
    for(auto it = graph.min_seq.begin(); it != graph.min_seq.end(); it++)
        cout << *it << " ";
    cout << endl;
    cout << "accu_reward: " << graph.accu_reward << endl;
    cout << "avg_reward: " << graph.avg_reward << endl;
    cout << "=============== Search Ended ===============" << endl;
    cout << endl << endl;
    #else
    cout << graph.searched_node_num << endl;
    #endif
}