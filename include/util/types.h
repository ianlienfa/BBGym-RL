#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <iostream>
#include <bitset>
#include <iomanip>
#include <random>
#include <limits>
#include "util/config.h"
#include <cassert>
using std::vector;
using std::bitset;

// typedefs
typedef vector<int> Vi;
typedef vector<Vi> VVi;
typedef bitset<JOB_NUMBER+1> B;
typedef vector<B> Vb;
typedef std::pair<double, double> Pdd;

// encodings
#define STATE_ENCODING vector<float>
#define ACTION_ENCODING int // 0: place, 1: insert + place, 2: left, 3: right

// typedef TIME_TYPE schedule_time;
#define BASIC_MAX(a,b) ((a)>(b)?(a):(b))
#define BASIC_MIN(a,b) ((a)<(b)?(a):(b))
#define BASIC_ABS(a) ((a)>0?(a):-(a))
#define BASIC_SWAP(a,b) {a^=b;b^=a;a^=b;}
#define assertm(msg, exp) assert(((void)msg, exp))

// Option Class Argument Wrapper
#define BBARG(type, name, init_val)\
    type _##name = (init_val);\
    void name(type val) {_##name = val;}\
    type& name() {return _##name;}

// Define Tracker Variable Wrapper
#define BB_TRACK_ARG(type, name, initval)\
    type _##name = initval;\
    bool name##_set = false;\
    type const & name() const {\
        return _##name;\
    }\
    type& name() {\
        if(!name##_set){\
            name##_set = true;\
            return _##name;\
        }\
        else {\
            assertm("Tracker Variable " #name " is already set!", false);\
        }\
    }
    
// Define Singleton Wrapper
#define SINGLETON(type, name)\
    static type* _##name;\
    static const type& get_##name() {\
        if (_##name == nullptr) {\
            _##name = new type();\
        }\
        return *_##name;\
    }\
    static void destroy() {\
        if (_##name != nullptr) {\
            delete _##name;\
            _##name = nullptr;\
        }\
    }    
#define SINGLETON_INIT(type, classname, name) type* classname::_##name = nullptr;

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

struct Measurer {
    std::chrono::steady_clock::time_point time_begin;
    std::chrono::steady_clock::time_point time_end;
    Measurer(){};
    void start() {
        time_begin = std::chrono::steady_clock::now();
    }
    void end(std::string msg = "") {
        time_end = std::chrono::steady_clock::now();
        std::cerr << std::setw(100) << msg << ": " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() << "ns" << std::endl;
    }
};

#if MEASURE_MODE == 1
    #define MEASURE(name, msg, command) Measurer name; name.start(); command name.end(msg);
#else 
    #define MEASURE(name, msg, command) command
#endif

// #define BB_RAND() (std::uniform_int_distribution<int64_t> dist(0, std::numeric_limits<int64_t>::max); dist(mt))
// #define BB_RANGED_RAND(min, max) (std::uniform_int_distribution<int64_t> dist((min), (max)); dist(mt))

extern std::mt19937 _bbgym_mt;
extern std::uniform_int_distribution<int> _bbgym_dist;
#define BB_RAND() _bbgym_dist(_bbgym_mt)

#endif