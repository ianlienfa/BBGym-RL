#ifndef ONE_RJ_SUM_CJ_NODE_H
#define ONE_RJ_SUM_CJ_NODE_H

#include "util/config.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
using std::vector;
using std::bitset;
using std::pair; using std::make_pair;
using std::cout; using std::endl;
using std::ostream;
using std::string;

#include "util/types.h"
#include "BB_engine/problemNode.h"

struct OneRjSumCjNode: ProblemNode
{
    struct NodeInfo{
        int id;
        TIME_TYPE earlest_start_time;
        TIME_TYPE completion_time;
        NodeInfo(){}
        NodeInfo(int id, TIME_TYPE earlest_start_time, TIME_TYPE completion_time):
            id(id), earlest_start_time(earlest_start_time), completion_time(completion_time){}
    };

    static string instance_name;
    static vector<TIME_TYPE> processing_time;
    static vector<TIME_TYPE> release_time;
    static vector<TIME_TYPE> job_weight;    
    static int jobs_num;
    static B jobs_mask;
    static float time_baseline;

    // node dependent data
    B is_processed;
    vector<int> seq;
    float lb;
    TIME_TYPE completion_time;    
    TIME_TYPE earliest_start_time;
    TIME_TYPE weighted_completion_time;

    // constructor
    OneRjSumCjNode();
    // basic constructor
    OneRjSumCjNode(B is_processed, vector<int> seq, float lb);
    // copy constructor
    OneRjSumCjNode(const OneRjSumCjNode& old);   

    int get_unfinished_jobs_num() const;
    bool bit_completion_test() const;
    friend ostream& operator<<(ostream& os, const OneRjSumCjNode& dt);
    vector<OneRjSumCjNode::NodeInfo> get_unfinished_jobs() const;
    void get_processing_time();

    // static functions        
    static bool cmpr(const OneRjSumCjNode &e1, const OneRjSumCjNode &e2){return e1.earliest_start_time < e2.earliest_start_time;}
    static pair<int, int> getObj(const vector<int> &seq); // return wjcj, cj
    static OneRjSumCjNode getInitESTSeq(); // return est, seq
    inline int get_earliest_st_time(int job_id) const;    
};

#endif