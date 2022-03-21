#ifndef BRANCH_MOD_H
#define BRANCH_MOD_H

#include "BB_engine/searchGraph.h"
#include "BB_engine/problemNode.h"

struct BranchMod
{
    // Inherited BranchMod should maintain its own graph and provide the following functions:
    // SearchGraph &graph;
    // void fill_graph(SearchGraph &graph){};
    // vector<ProblemNode> branch(ProblemNode current_node){};
};


#endif