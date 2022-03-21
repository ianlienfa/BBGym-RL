#ifndef JobWithReleaseDate_h
#define JobWithReleaseDate_h

// standard library
#include <iostream>

typedef struct JobWithReleaseDate
{
    int r;
    int idx;
    int p;
    JobWithReleaseDate(){}
    JobWithReleaseDate(int idx, int p, int r = 0){this->r = r; this->idx = idx; this->p = p;}
    friend ostream& operator<<(ostream &out, const JobWithReleaseDate &j){
        return out << "(idx: " << j.idx << ", p: " << j.p << ", r: " << j.r << ")";
    }
    friend bool operator<(const JobWithReleaseDate &jr1, const JobWithReleaseDate& jr2){return jr1.p < jr2.p;}
    friend bool operator>(const JobWithReleaseDate &jr1, const JobWithReleaseDate& jr2){return jr1.p > jr2.p;}
} Jr;

# endif