#ifndef SEARCH_MOD_H
#define SEARCH_MOD_H

#include <iostream>
#include <vector>
using std::vector;

#include "BB_engine/searchGraph.h"
#include "BB_engine/problemNode.h"

struct SearchMod
{
private:
    bool graph_filled = false;
    bool find_optimal = false;
public:
    
    // Inherited SearchMod should maintain its own graph and provide the following functions:
    // SearchGraph &graph;
    // SearchGraph init(ProblemNode rootProblem){};
    // ProblemNode search_next(){};
    // void update_graph(ProblemNode &current_node, vector<ProblemNode>& branched_nodes){};    
    // void fill_graph(SearchGraph &graph){};

    virtual double get_optimal(){return 0.0;};
    virtual bool get_find_optimal(){return false;};
};


#endif