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
#include <chrono>
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
#define ACTION_ENCODING vector<float> // 0001: place, 0010: insert + place, 0100: left, 1000: right

// typedef TIME_TYPE schedule_time;
#define BASIC_MAX(a,b) ((a)>(b)?(a):(b))
#define BASIC_MIN(a,b) ((a)<(b)?(a):(b))
#define BASIC_ABS(a) ((a)>0?(a):-(a))
#define BASIC_SWAP(a,b) {a^=b;b^=a;a^=b;}
#define assertm(msg, exp) assert(((void)msg, exp))

// Option Class Argument Wrapper
#define BBARG(parent, type, name, init_val)\
    type _##name = (init_val);\
    parent& name(type val) {_##name = val; return *this;}\
    type& name() {return _##name;}

// define arguments that has extra things to do when setting
#define BB_FUNC_ARG(parent, type, name, init_val, setfunc)\
    type _##name = (init_val);\
    parent& name(type val) {_##name = val; cout << "running setfunc" << endl; setfunc(val); return *this;}\
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
            return _##name;\
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

#define GAME_TRACKER 0
#if GAME_TRACKER == 1
    #define GAME_TRACK(name, command) cout << "==========" << endl << name << "() is called, " << "picker pos: " << picker_pos << ", current pos: " << current_pos << endl; print(); command cout << "---------- " << endl; print(); cout << name <<  " done" << endl << "==========" << endl; 
#else
    #define GAME_TRACK(name, command) command
#endif


extern std::mt19937 _bbgym_mt;
extern std::uniform_int_distribution<int> _bbgym_dist;
#define BB_RAND() _bbgym_dist(_bbgym_mt)

#endif