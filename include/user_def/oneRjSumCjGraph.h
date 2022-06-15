#ifndef ONE_RJ_SUM_CJ_GRAPH_H
#define ONE_RJ_SUM_CJ_GRAPH_H

#include <vector>
#include <iomanip>
#include <queue>
#include <map>
#include <tuple>
#include <limits>
#include <memory>
using std::vector;
using std::map;
using std::cerr;
using std::numeric_limits;

#include "BB_engine/searchGraph.h"
#include "util/types.h"
#include "util/PriorityQueue.h"
#include "util/PlacementList.h"
#include "util/config.h"
#include "user_def/oneRjSumCjNode.h"
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
#elif (SEARCH_STRATEGY == searchOneRjSumCj_CBFS_LIST)
    PlacementList<OneRjSumCjNode, int64_t> contours;
#endif

    // dynamic during search
    Vi min_seq;
    double min_obj;
    OneRjSumCjNode current_node;
    bool optimal_found;
    float accu_reward;
    float avg_reward;

    OneRjSumCjGraph(){
        optimal_found = false;
        searched_node_num = 0;
        accu_reward = 0;
        avg_reward = 0;
    };
    OneRjSumCjGraph(vector<TIME_TYPE> &processing_time, vector<TIME_TYPE> &release_time, vector<TIME_TYPE> &job_weight, int jobs_num): OneRjSumCjGraph()
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
#elif (SEARCH_STRATEGY == searchOneRjSumCj_CBFS_LIST)
        contours.init(OneRjSumCjNode::cmpr);        
#endif
        min_seq = Vi();
        min_obj = numeric_limits<double>::max();        
    }

#if(SEARCH_STRATEGY == searchOneRjSumCj_CBFS_LIST)
    vector<float> get_contour_snapshot() const
    {
        return contours.get_snapshot();
    }    
    OneRjSumCjGraph& set_max_size(int max_num_contour){
        contours.set_max_size(max_num_contour);
        return *this;
    }
#elif (SEARCH_STRATEGY == searchOneRjSumCj_CBFS)    
    vector<float> get_contour_snapshot(int max_num_contour) const
    {
        const float norm_factor = 1e3;
        vector<float> contour_snapshot;
        assertm("contour size exceeds max_num_contour", max_num_contour >= contours.size());
        contour_snapshot.assign(max_num_contour, -1e-11);      
        int i = 0;  
        for(const auto &iter: contours){
            if((!iter.second.empty()) && (iter.first != 0.0)){
                contour_snapshot[i] = ((float)(iter.second.size())) * iter.first / norm_factor;
                assertm("snapshot with zero value", contour_snapshot[i] != 0.0);
                i++;
            }    
        }        
        return contour_snapshot;
    }
    // return the contour configuration (labels) and the current iter
    std::tuple<std::unique_ptr<vector<int>>, int> get_contour_config(int max_num_contour) const
    {
        std::unique_ptr<vector<int>> contour_config_ptr = std::make_unique<vector<int>>();
        vector<int> &contour_config = *contour_config_ptr;
        assertm("contour size exceeds max_num_contour", max_num_contour >= contours.size());        
        int current_contour_pointer = 0;  
        for(auto iter = contours.begin(); iter != contours.end(); iter++){
            contour_config.push_back(iter->first);
            if(iter == current_contour_iter){
                current_contour_pointer = contour_config.size() - 1;
            }
        }        
        return make_tuple(std::move(contour_config_ptr), current_contour_pointer);
    }
    // clip inserting for rnn, with max_num_contour
    void clip_insert(int max_num_contour, map<CONTOUR_TYPE, PriorityQueue<OneRjSumCjNode>> &contour, OneRjSumCjNode &node, float label)
    {
        if(contour.size() < max_num_contour)
        {
            assertm("label_decision(): label is out of range", (label > 0) && (label < 5));
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
            assertm("should not push into 0 contour: ", target_contour_iter->first != 0);
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