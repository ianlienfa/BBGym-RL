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
// #if (Labeling_Strategy == RL_DDPR)
//     #include "search_modules/Net/DDPR/NetDDPR.h"
// #endif
struct OneRjSumCjGraph: SearchGraph
{
    // static during search
    vector<TIME_TYPE> processing_time;
    vector<TIME_TYPE> release_time;
    vector<TIME_TYPE> job_weight;
    int jobs_num;
    long long searched_node_num;

#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
    map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>> contours;
    map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>::iterator current_contour_iter;
#elif (SEARCH_STRATEGY == searchOneRjSumCj_LU_AND_SAL)
    vector<deque<OneRjSumCjNode>> contours;
    int current_level;
#endif

    // dynamic during search
    Vi min_seq;
    double min_obj;
    OneRjSumCjNode current_node;
    bool optimal_found;
    float accu_reward;

    OneRjSumCjGraph(){
        optimal_found = false;
        searched_node_num = 0;
        accu_reward = 0;
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
        contours = map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>();
        current_contour_iter = contours.begin();
#endif
        min_seq = Vi();
        min_obj = numeric_limits<double>::max();        
    }
#if (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)
    vector<float> get_contour_snapshot(int max_num_contour)
    {
        vector<float> contour_snapshot;
        contour_snapshot.assign(max_num_contour, 0);      
        int i = 0;  
        for(auto iter = contours.begin(); iter != contours.end(); iter++, i++){
            contour_snapshot[i] = iter->second.size() * iter->first;
        }        
        return contour_snapshot;
    }
    // clip inserting for rnn, with max_num_contour
    void clip_insert(int max_num_contour, map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>> &contour, OneRjSumCjNode &node, int label)
    {
        if(contour.size() < max_num_contour)
        {
            map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>::iterator target_contour_iter = contour.find(label);
            if(target_contour_iter == contours.end())
            {
                PriorityQueue<OneRjSumCjNode> pq_insert(OneRjSumCjNode::cmpr);
                pq_insert.push(node);
                #if DEBUG_LEVEL >=2
                    cout << "inserting node to contour " << label << ": " << *it  << endl;
                #endif
                contours.insert(make_pair(label, pq_insert));
            }
            else
            {
                target_contour_iter->second.push(node);
                #if DEBUG_LEVEL >=2
                    cout << "inserting node to contour " << label << ": " << *it  << endl;
                #endif
            }      
        }  
        else if(contour.size() == max_num_contour)
        {
            // put into the closest contour
            map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>>::iterator target_contour_iter = contour.lower_bound(label);
            if(target_contour_iter == contours.end())
            {
                target_contour_iter--;
            }
            target_contour_iter->second.push(node);
        }
        else
        {
            assertm("Error: contour.size() > max_num_contour", false);
        }
    }
#endif

};

#endif