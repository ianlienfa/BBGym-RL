#ifndef ONE_RJ_SUM_CJ_BRANCH_H
#define ONE_RJ_SUM_CJ_BRANCH_H

#include "branch_modules/branchMod.h"
#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"
#include "util/LowerBound.h"

struct OneRjSumCjBranch: BranchMod
{
    OneRjSumCjGraph *graph;

    void fill_graph(OneRjSumCjGraph *graph);
    vector<OneRjSumCjNode> branch(const OneRjSumCjNode &current_node, LowerBound &lower_bound);
};


#endif