#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <iostream>
#include <bitset>
#include "util/config.h"
using std::vector;
using std::bitset;

// typedefs
typedef vector<int> Vi;
typedef vector<Vi> VVi;
typedef bitset<JOB_NUMBER+1> B;
typedef vector<B> Vb;
typedef std::pair<double, double> Pdd;

// typedef TIME_TYPE schedule_time;
#define BASIC_MAX(a,b) ((a)>(b)?(a):(b))
#define BASIC_MIN(a,b) ((a)<(b)?(a):(b))
#define BASIC_ABS(a) ((a)>0?(a):-(a))
#define BASIC_SWAP(a,b) {a^=b;b^=a;a^=b;}

// exception class
class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};

// operator overloading
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
        os << " " << *ii;
    }
    os << " ]";
    return os;
}

#endif