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
        OneRjSumCjNode::worst_upperbound += OneRjSumCjNode::processing_time[i];
        OneRjSumCjNode::worst_upperbound += OneRjSumCjNode::release_time[i];
    }
    cin >> tmp;
    if(tmp != -1){ 
        cout << "input error" << endl;
        return false;
    }
    return true;
}

bool parse_and_init_oneRjSumCj(const string& file_name)
{   
    ifstream fin;
    fin.open(file_name);

    // Read the number of jobs
    int tmp;
    fin >> OneRjSumCjNode::jobs_num;
    OneRjSumCjNode::instance_name = file_name;
    OneRjSumCjNode::processing_time.assign(OneRjSumCjNode::jobs_num+1, 0);
    OneRjSumCjNode::release_time.assign(OneRjSumCjNode::jobs_num+1, 0);
    OneRjSumCjNode::job_weight.assign(OneRjSumCjNode::jobs_num+1, 0);
    OneRjSumCjNode::jobs_mask = B(0).set().reset(0);
    

    // construct
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){fin >> OneRjSumCjNode::processing_time[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){fin >> OneRjSumCjNode::release_time[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){fin >> OneRjSumCjNode::job_weight[i]; } 
    for(int i = 1; i <= OneRjSumCjNode::jobs_num; i++){
        OneRjSumCjNode::worst_upperbound += OneRjSumCjNode::processing_time[i];
        OneRjSumCjNode::worst_upperbound += OneRjSumCjNode::release_time[i];
    }
    fin >> tmp;
    if(tmp != -1){ 
        cout << file_name << ": input error" << endl;
        return false;
    }
    fin.close();
    return true;
}
