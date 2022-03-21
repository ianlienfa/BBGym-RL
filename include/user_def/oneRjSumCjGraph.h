#ifndef ONE_RJ_SUM_CJ_GRAPH_H
#define ONE_RJ_SUM_CJ_GRAPH_H

#include <vector>
#include <queue>
#include <map>
#include <limits>
using std::vector;
using std::map;
using std::cerr;
using std::numeric_limits;

#include "BB_engine/searchGraph.h"
#include "util/types.h"
#include "util/PriorityQueue.h"
#include "util/config.h"

struct OneRjSumCjGraph: SearchGraph
{
    // static during search
    vector<TIME_TYPE> processing_time;
    vector<TIME_TYPE> release_time;
    vector<TIME_TYPE> job_weight;
    int jobs_num;
    long long searched_node_num;

#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
    map<int, PriorityQueue<OneRjSumCjNode>> contours;
    map<int, PriorityQueue<OneRjSumCjNode>>::iterator current_contour_iter;
#elif (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
    vector<deque<OneRjSumCjNode>> contours;
    int current_level;
#endif

    // dynamic during search
    Vi min_seq;
    double min_obj;
    OneRjSumCjNode current_node;
    bool optimal_found;

    OneRjSumCjGraph(){
        optimal_found = false;
        searched_node_num = 0;
    };
    OneRjSumCjGraph(vector<TIME_TYPE> &processing_time, vector<TIME_TYPE> &release_time, vector<TIME_TYPE> &job_weight, int jobs_num)
    {
        this->processing_time = processing_time;
        this->release_time = release_time;
        this->job_weight = job_weight;
        this->jobs_num = jobs_num;
        optimal_found = false;
        searched_node_num = 0;

        if(!(processing_time.size() == release_time.size()) || jobs_num <= 0){
            cerr << "Error: processing_time.size() != release_time.size() || jobs_num <= 0" << endl;
            exit(INVALID_INPUT);
        }
#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
        contours = map<int, PriorityQueue<OneRjSumCjNode>>();
        current_contour_iter = contours.begin();
#endif
        min_seq = Vi();
        min_obj = numeric_limits<double>::max();        
    }
};

#endif