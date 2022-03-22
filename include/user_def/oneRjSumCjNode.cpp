#include "oneRjSumCjNode.h"

vector<TIME_TYPE> OneRjSumCjNode::processing_time;
vector<TIME_TYPE> OneRjSumCjNode::release_time;
vector<TIME_TYPE> OneRjSumCjNode::job_weight;
int OneRjSumCjNode::jobs_num;
B OneRjSumCjNode::jobs_mask;
float OneRjSumCjNode::worst_upperbound;
string OneRjSumCjNode::instance_name;

OneRjSumCjNode::OneRjSumCjNode()
{
    is_processed = B(0);
    seq = vector<int>();
    lb = 0;
    completion_time = 0.0;
    earliest_start_time = 0.0;
    weighted_completion_time = 0.0;
};

OneRjSumCjNode::OneRjSumCjNode(const OneRjSumCjNode& old) 
{
    this->is_processed = old.is_processed;
    this->seq = old.seq;
    this->lb = old.lb;
    this->completion_time = old.completion_time;
    this->earliest_start_time = old.earliest_start_time;
    this->weighted_completion_time = old.weighted_completion_time;
}

OneRjSumCjNode::OneRjSumCjNode(B is_processed, vector<int> seq, float lb) : OneRjSumCjNode()
{
    this->is_processed = is_processed;
    this->seq = seq;
    this->lb = lb;
}

ostream& operator<<(ostream& os, const OneRjSumCjNode& dt)
{
    os << "{ ";
    os << "is_processed: " << dt.is_processed << ", ";
    os << "seq: ";
    for(size_t i = 0; i < dt.seq.size(); i++){
        os << dt.seq[i] << " ";
    }
    os << ", ";
    os << "lb: " << dt.lb << ", ";
    os << "completion_time: " << dt.completion_time << ", ";
    os << "earliest_start_time: " << dt.earliest_start_time << ", ";
    os << "weighted_completion_time: " << dt.weighted_completion_time << ", ";
    os << "}";
    return os;
}

bool OneRjSumCjNode::bit_completion_test() const {
    return (is_processed ^ OneRjSumCjNode::jobs_mask).none();
}


int OneRjSumCjNode::get_earliest_st_time(int job_id) const {
    // if(completion_time == 0.0)
    //     exit(NOT_INIT);
    return BASIC_MAX(release_time[job_id], completion_time);
}

// <SigmaWjCj, SigmaCj>
pair<int, int> OneRjSumCjNode::getObj(const vector<int> &seq) {
    int rj = 0;
    int Cj = 0;
    int WjCj = 0;
    for(size_t i = 0; i < seq.size(); i++)
    {        
        int job_id = seq[i];
        rj =  BASIC_MAX(Cj, OneRjSumCjNode::release_time[job_id]);
        Cj =  rj + OneRjSumCjNode::processing_time[job_id];
        WjCj += OneRjSumCjNode::job_weight[job_id] * Cj;
        #if DEBUG_LEVEL >= 3
        cout << "r[" << job_id << "] = " << rj << endl;
        cout << "WjCj: " << WjCj << endl;
        cout << "Cj: " << Cj << endl;
        #endif
    }   
    int weighted_completion_time = WjCj;
    int completion_time = Cj;
    return make_pair(weighted_completion_time, completion_time);
}

OneRjSumCjNode OneRjSumCjNode::getInitESTSeq() {
    OneRjSumCjNode node;
    node.seq = vector<int>();
    node.completion_time = 0.0;
    vector<pair<int, int>> ESTseq; // <id, EST>
    vector<int> seq;
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++)
        ESTseq.push_back(make_pair(i, OneRjSumCjNode::release_time[i]));
    sort(ESTseq.begin(), ESTseq.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
        return a.second < b.second;
    });
    for(size_t i = 0; i < ESTseq.size(); i++)
        node.seq.push_back(ESTseq[i].first);
    pair<int, int> incumbent = getObj(node.seq);
    node.weighted_completion_time = incumbent.first;
    node.completion_time = incumbent.second;
    return node;
}

int OneRjSumCjNode::get_unfinished_jobs_num() const {
    int bitset_job_done_num = is_processed.count();
    #if VALIDATION_LEVEL == validation_level_HIGH
    // check coherence of the bitset and seq
    int seq_job_done_num = seq.size();
    if(bitset_job_done_num != seq_job_done_num)
    {
        cout << "bitset_job_done_num:" << bitset_job_done_num << "!=  seq_job_done_num:" << seq_job_done_num << "!" << endl;
        exit(LOGIC_ERROR);
    }
    #endif
    return jobs_num - bitset_job_done_num;
}

vector<OneRjSumCjNode::NodeInfo> OneRjSumCjNode::get_unfinished_jobs() const {
    vector<OneRjSumCjNode::NodeInfo> unfinished_jobs;
    for(int i = 1; i <= jobs_num; i++)
    {        
        if(!is_processed[i])
        {
            TIME_TYPE st= get_earliest_st_time(i);        
            #if DEBUG_LEVEL >= 3
            cout << "st: " << st << endl;
            #endif
            unfinished_jobs.push_back(OneRjSumCjNode::NodeInfo(i, st, st+processing_time[i]));
        }
    }        
    return unfinished_jobs;
}

void OneRjSumCjNode::get_processing_time() {
    for(auto it: processing_time)
        cout << it << " ";
    cout << endl;
}