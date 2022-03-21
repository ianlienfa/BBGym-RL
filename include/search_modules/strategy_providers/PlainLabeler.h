#ifndef PLAIN_LABELER_H
#define PLAIN_LABELER_H

#include "user_def/oneRjSumCjNode.h"
#include "search_modules/strategy_providers/Labeler.h"

struct PlainLabeler: Labeler
{
    PlainLabeler();
    int operator()(const OneRjSumCjNode &node) const;    
};


#endif