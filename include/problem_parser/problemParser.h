#ifndef PROBLEM_PARSER_H
#define PROBLEM_PARSER_H

// Standard Library and others
#include "util/config.h"
#include "util/types.h"
#include "user_def/oneRjSumCjNode.h"

#include <iostream>
#include <fstream>
#include <filesystem>
using std::cout; using std::endl;
using std::cin;
using std::ifstream;

struct InputHandler
{   
    string path;
    std::filesystem::directory_iterator current_file;
    vector<string> file_list;
    vector<string>::iterator file_list_it;
    
    InputHandler(string path)
    {
        this->path = path;
        current_file = std::filesystem::directory_iterator(path);
    }
    void fill_file_list()
    {
        for (const auto & entry : current_file)
        {
            if(entry.is_regular_file())
            {
                file_list.push_back(entry.path());
            }            
        }
    }
    string getNextFileName() // ignore file if not regular and go the start if end is meet
    {
        if(file_list.empty())
        {
            fill_file_list();
            file_list_it = file_list.begin();
            return *file_list_it;
        }
        else
        {            
            string path = *file_list_it;
            file_list_it++;            
            if(file_list_it == file_list.end())
            {
                file_list_it = file_list.begin();
            }
            return path;
        }
    }
    void reset()
    {
        current_file = std::filesystem::directory_iterator(path);
    }
};
bool parse_and_init_oneRjSumCj();
bool parse_and_init_oneRjSumCj(const string& file_name);



#endif