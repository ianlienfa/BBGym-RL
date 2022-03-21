#include "problemParser.h"

bool parse_and_init_oneRjSumCj()
{   
    // Read the number of jobs
    int tmp;
    cin >> OneRjSumCjNode::jobs_num;
    OneRjSumCjNode::processing_time.resize(OneRjSumCjNode::jobs_num+1);
    OneRjSumCjNode::release_time.resize(OneRjSumCjNode::jobs_num+1);
    OneRjSumCjNode::job_weight.resize(OneRjSumCjNode::jobs_num+1);
    OneRjSumCjNode::jobs_mask = B(0).set().reset(0);
    

    // construct
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){cin >> OneRjSumCjNode::processing_time[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){cin >> OneRjSumCjNode::release_time[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){cin >> OneRjSumCjNode::job_weight[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){
        OneRjSumCjNode::time_baseline += OneRjSumCjNode::processing_time[i];
        OneRjSumCjNode::time_baseline += OneRjSumCjNode::release_time[i];
    }
    cin >> tmp;
    if(tmp != -1){ 
        cout << "input error" << endl;
        return false;
    }
    return true;
}
