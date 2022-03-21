#include "util/LowerBound.h"

#if LOWER_BOUND == lowerbound_oneRjSumCj_LU_AND_SAL
double LowerBound::operator()(const OneRjSumCjNode &node){
    double lb = node.weighted_completion_time;
    vector<OneRjSumCjNode::NodeInfo> unfinished_nodeinfo = node.get_unfinished_jobs();
    for(auto it : unfinished_nodeinfo){
        lb += it.completion_time * OneRjSumCjNode::job_weight[it.id];
    }
    int lap = 0;
    int unfinished_job_num = unfinished_nodeinfo.size();
    for(int i = 0; i < unfinished_job_num-1; i++)
    {
        for(int j = i+1; j < unfinished_job_num; j++)
        {
            #if DEBUG_LEVEL >= 3
            cout << "j: " << j << endl; 
            #endif
            int back_id = i;
            int front_id = j;
            if(unfinished_nodeinfo[i].completion_time < unfinished_nodeinfo[j].completion_time)
                BASIC_SWAP(back_id, front_id);
            lap = OneRjSumCjNode::processing_time[unfinished_nodeinfo[back_id].id] + OneRjSumCjNode::processing_time[unfinished_nodeinfo[front_id].id] - BASIC_ABS(unfinished_nodeinfo[back_id].completion_time - unfinished_nodeinfo[front_id].earlest_start_time);
            if(lap > 0)
            {
                #if DEBUG_LEVEL >= 3
                cout << "[" << endl;
                cout << unfinished_nodeinfo[back_id].id << "-- est: " << unfinished_nodeinfo[back_id].earlest_start_time << ", cpt: " << unfinished_nodeinfo[back_id].completion_time << endl;
                cout << unfinished_nodeinfo[front_id].id << "-- est: " << unfinished_nodeinfo[front_id].earlest_start_time << ", cpt: " << unfinished_nodeinfo[front_id].completion_time << endl;
                cout << "lap" << lap << endl;
                cout << "]" << endl;
                #endif
                lb += lap * BASIC_MIN(OneRjSumCjNode::job_weight[unfinished_nodeinfo[back_id].id], OneRjSumCjNode::job_weight[unfinished_nodeinfo[front_id].id]);
            }
            
        }
    }    
    return lb;
}
#endif

/*----------------------------------------------------------------------------
 * Function: SRPT (lower bound)
 * Usage: SRPT(E);
 ----------------------------------------------------------------------------
 - The overload version of SRPT for PartialScedule
 ----------------------------------------------------------------------------*/
int LowerBound::SRPT(const OneRjSumCjNode &e) {
    cout << "[LowerBound::SRPT] not yet implemented!" << endl;
    exit(-1);
}


/*----------------------------------------------------------------------------
 * Function: SRPT (lower bound)
 * Usage: SRPT(qu);
 ----------------------------------------------------------------------------
 - qu is a 2d queue with release date, 
    the job with same release date "should be save in the same row"
 ----------------------------------------------------------------------------*/
int LowerBound::SRPT(QQjr qu, bool debug)
{
    struct seqJ{
        int p;
        int ci;
        int idx;
        seqJ(int idx, int p, int ci){this->idx = idx; this->p = p; this->ci = ci;}
    };
    vector<seqJ> seq;

    // priority queue init
    bool (*cmpr)(const Jr &jr1, const Jr& jr2) = [](const Jr &jr1, const Jr& jr2){return jr1.p < jr2.p;};
    PriorityQueue<Jr> pq(cmpr);
    if(!qu.size()) return 0;
    int t = qu[0][0].r;
    int crtcl = qu[0][0].r;

    // iterations
    do
    {
        // push job at current time
        Qjr jobs;
        if(qu.size())
        {
            jobs = qu.front();  // get the earliest releasing jobs
            if (jobs.size()) {
                if (debug) cout << "add: " << endl;
                if (t == jobs[0].r) // if current time matches the release time, push these jobs into pq
                {
                    qu.pop_front();
                    for (Qjr::iterator it = jobs.begin(); it != jobs.end(); it++) {
                        pq.push(*it);    
                        if(debug) cout << *it << endl;                   
                    }
                }
            }
        }

        // priority queue inspection
        if (debug) 
        {
            cout << "pq snapshot: " << endl;
            pq.bst_print();
            cout << "extract: " << pq.top() << endl;
            cout << endl;
        }

        Jr leading;
        if(pq.size())
        {
            // pop the leading job from priority queue
            leading = pq.extract();
            // if(debug) cout << "leading: " << leading << endl;

            // update critical time point
            crtcl = (qu.size()) ? min(qu.front()[0].r, t + leading.p) : t + leading.p;

            // update vec and heap(if needed)
            int ci = crtcl;
            int pi = crtcl - t;
            int less = leading.p + t - crtcl;
            if(less)
                pq.push(Jr(leading.idx, less));
            else
                seq.push_back(seqJ(leading.idx, pi, ci));
        }
        else
        {
            // update critical time point
            if(qu.size())
                crtcl = qu.front()[0].r;
        }
        
        if(debug) cout << "crtrl: " << crtcl << endl;

        // update current time
        t = crtcl;

    }while(qu.size() || pq.size());

    // after calculation, compute sigma Cj
    int sigmaCj = 0;
    for(int i = 0; i < seq.size(); i++)
    {
        sigmaCj += seq[i].ci;

        // print for debug
        if(debug)
        printf("(idx: %d, Pi: %d, Ci: %d), ", seq[i].idx, seq[i].p, seq[i].ci);
    }
    if(debug) printf("\n");

    if(debug) cout << "sigmaCj: " << sigmaCj << endl;
    return sigmaCj;
}

int LowerBound::SRPT(QQjr &qu)
{
    return SRPT(qu, false);
}

void LowerBound::lowerBoundPrint()
{
    cout << "yes lowerbound" << endl;
}


