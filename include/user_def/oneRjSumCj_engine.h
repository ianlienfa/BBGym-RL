#ifndef ONE_RJ_SUM_CJ_ENGINE_H
#define ONE_RJ_SUM_CJ_ENGINE_H

// STL includes
#include <memory>
using std::shared_ptr;

// dependencies 
#include "search_modules/strategy_providers/PlainLabeler.h"
#include "user_def/oneRjSumCjGraph.h"
#include "user_def/oneRjSumCjSearch.h"
#include "user_def/oneRjSumCjBranch.h"
#include "user_def/oneRjSumCjPrune.h"
#include "user_def/oneRjSumCjNode.h"
#include "util/types.h"
#include "util/config.h"
#include "util/LowerBound.h"
#include "util/PriorityQueue.h"

/*--------------------------------------------------------------------------
 The Engine knows about the problem:
     does the incumbent update and objective computation

 The Searcher maintain the search tree:
     provide info about the current node, and the next node to be explored

 The Brancher looks at the current node and return the branched nodes

 The Pruner looks at the branched nodes and decide which nodes to remove
 
 * update_incumbent() is a problem-specific function that updates the incumbent
   solution and seqence based on the current node.

--------------------------------------------------------------------------*/


struct OneRjSumCj_engine
{    
    OneRjSumCjGraph graph;
    OneRjSumCjSearch searcher;
    OneRjSumCjBranch brancher;
    OneRjSumCjPrune pruner;
    OneRjSumCjNode rootProblem; 
    LowerBound lowerbound;    
    double optimal;
    OneRjSumCj_engine();
    OneRjSumCj_engine(OneRjSumCjGraph graph, OneRjSumCjSearch searcher, OneRjSumCjBranch brancher, OneRjSumCjPrune pruner, LowerBound lowerbound);
    OneRjSumCjGraph solve(OneRjSumCjNode rootProblem);    
    void update_incumbent(const OneRjSumCjNode &current_node);
    TIME_TYPE objSolve(const OneRjSumCjNode &current_node);
    
    double get_optimal(){};
    double get_jobs_num();
};


void print_config();
void post_print_config(const OneRjSumCjGraph &graph);


#endif