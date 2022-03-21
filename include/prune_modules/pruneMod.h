#ifndef PRUNE_MOD_H
#define PRUNE_MOD_H

/*----------------------------------------------------------------------------
 * Struct: PruneMod
 * Specification: The struct of the prune module
 * Usage: 
 * Constructor: 
------------------------------------------------------------------------------
 * prune() prvides basic prune function iterative calls, 
   and should also be call in the override version 
----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>

#include "BB_engine/searchGraph.h"
#include "BB_engine/problemNode.h"
#include "util/config.h"
using std::vector;

struct PruneMod
{
    // Inherited PruneMod should maintain its own graph and provide the following functions:
    // SearchGraph &graph;
    // void fill_graph(SearchGraph &graph);
    // void fill_prune_funcs(vector<prune_func> &prune_funcs);
    
    template <typename T>
    vector<T> prune(vector<T>& branched_nodes, vector<vector<T>(*)(vector<T>&)> prune_funcs);
};

#endif