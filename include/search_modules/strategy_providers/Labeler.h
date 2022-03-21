#ifndef LABELER_H
#define LABELER_H

#include "user_def/oneRjSumCjNode.h"

struct Labeler
{
    Labeler(){}
    virtual int operator()(const OneRjSumCjNode &node) const = 0;
};


#endif