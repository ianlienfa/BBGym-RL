#include "pruneMod.h"

template <typename T>
vector<T> PruneMod::prune(vector<T>& branched_nodes, vector<vector<T>(*)(vector<T>&)> prune_funcs)
{
    if(prune_funcs.empty())
    {
        cerr << "[NOT_INIT] Error: prune_funcs is empty" << endl;
        exit(NOT_INIT);
    }
    for(auto it = prune_funcs.begin(); it != prune_funcs.end(); it++)
    {
        branched_nodes = (*it)(branched_nodes);
    }
    return branched_nodes;
}

// void PruneMod::fill_graph(SearchGraph &graph)
// {
//     this->graph = graph;
// }

// void PruneMod::fill_prune_funcs(vector<prune_func> &prune_funcs)
// {
//     this->prune_funcs = prune_funcs;
// }