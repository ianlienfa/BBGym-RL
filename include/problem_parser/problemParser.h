#ifndef PROBLEM_PARSER_H
#define PROBLEM_PARSER_H

// Standard Library and others
#include "util/config.h"
#include "util/types.h"
#include "user_def/oneRjSumCjNode.h"

#include <iostream>
#include <filesystem>
using std::cin; using std::cout; using std::endl;
using std::string;

struct InputHandler
{   
    string path;
    std::filesystem::directory_iterator current_file;
    
    InputHandler(string path)
    {
        this->path = path;
        current_file = std::filesystem::directory_iterator(path);
    }
    string getCurrentFileName()
    {
        return string(current_file->path());
    }
    string getNextFileName()
    {
        current_file++;
        if(current_file != std::filesystem::end(std::filesystem::directory_iterator(path)))
        {
            string filename = string(current_file->path());
            return filename;
        }
        return "";
    }
};

bool parse_and_init_oneRjSumCj();



#endif