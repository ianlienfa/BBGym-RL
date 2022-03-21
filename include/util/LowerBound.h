#ifndef LOWERBOUND_H
#define LOWERBOUND_H

#include "user_def/oneRjSumCjNode.h"
#include "PriorityQueue.h"
#include "JobWithReleaseDate.h"

// typedef
typedef deque<Jr> Qjr;
typedef deque<Qjr> QQjr;

struct LowerBound {
    static int SRPT(QQjr &qu);
    static int SRPT(QQjr qu, bool debug);
    static int SRPT(const OneRjSumCjNode &e);
    static void lowerBoundPrint();
    double operator()(const OneRjSumCjNode &e);
};

#endif