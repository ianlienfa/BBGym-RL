#ifndef ONE_RJ_SUM_CJ_PRUNE_H
#define ONE_RJ_SUM_CJ_PRUNE_H

#include <iostream>
#include <vector>
#include <algorithm>    
using std::vector;

#include "prune_modules/pruneMod.h"
#include "user_def/oneRjSumCjNode.h"
#include "user_def/oneRjSumCjGraph.h"


typedef void (*prune_func_pt)(vector<OneRjSumCjNode>&, OneRjSumCjGraph &graph);
typedef void (*prune_func_safept)(vector<OneRjSumCjNode>&, const OneRjSumCjGraph &graph);

struct OneRjSumCjPrune: public PruneMod {

    OneRjSumCjGraph *graph; 
    static vector<prune_func_pt> prune_funcs;
    static vector<prune_func_safept> safe_prune_funcs;
    void fill_graph(OneRjSumCjGraph *graph);
    void prune(vector<OneRjSumCjNode> &branched_nodes);
    void prune_func_init(vector<prune_func_pt>&);
    void prune_func_init();
    void prune_update_incumbent(Vi min_seq, int min_obj);    
    // if custum prune function is needed, add it here
    // ...

};

// prune_functions
// vector<OneRjSumCjNode> pruneBianco1982_1(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_2(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_3(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_4(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_5(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_6(vector<OneRjSumCjNode>&){};
// vector<OneRjSumCjNode> pruneBianco1982_7(vector<OneRjSumCjNode>&){};
void pruneIncumbentCmpr(vector<OneRjSumCjNode>&, const OneRjSumCjGraph &graph);
void prune__OneRjSumCj__LU_AND_SAL__Theorem1(vector<OneRjSumCjNode>& branched_nodes, OneRjSumCjGraph &graph);

#endif