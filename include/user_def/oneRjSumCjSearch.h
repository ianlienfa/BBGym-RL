#ifndef ONE_RJ_SUM_CJ_SEARCH_H
#define ONE_RJ_SUM_CJ_SEARCH_H

#include <iostream>
#include <vector>
#include <utility>
using std::vector;
using std::cerr;
using std::make_pair;

#include "search_modules/searchMod.h"
#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"
#include "search_modules/strategy_providers/PlainLabeler.h"
#include "util/LowerBound.h"

struct OneRjSumCjSearch: SearchMod
{
    OneRjSumCjGraph *graph;

#if SEARCH_STRATEGY == searchOneRjSumCj_CBFS        
    PlainLabeler labeler;
#endif

    bool get_find_optimal();    
    void fill_graph(OneRjSumCjGraph *graph);
    void history_fill();
    vector<OneRjSumCjNode> update_graph(OneRjSumCjNode current_node, vector<OneRjSumCjNode> branched_nodes);
    OneRjSumCjGraph init(OneRjSumCjNode rootProblem);
    OneRjSumCjNode search_next();
    bool is_incumbent(const OneRjSumCjNode &current_node);
};


#endif