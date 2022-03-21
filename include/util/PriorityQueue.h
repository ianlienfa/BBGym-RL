//
// Created by 林恩衍 on 2021/6/28.
//

#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H

#include <iostream>
#include <vector>
#include <queue>
using std::vector;
using std::deque;
using std::cout; using std::endl;

#include "config.h"

template <class T>
class PriorityQueue
{
private:
    int sz;
    bool debug = false;
    vector<T> arr;
    bool (*cmpr)(const T& t1, const T &t2);
    void swap(T &t1, T &t2){T tmp = t1; t1 = t2; t2 = tmp;}
    void heapify(int i);
    inline int left(int i){return i * 2;}
    inline int right(int i){return i * 2 + 1;}
    inline int parent(int i){return i / 2;}
    void sizeInit(vector<T> v){
        arr.resize(v.size()+1);
        copy(v.begin(), v.end(), arr.begin()+1);    // make arr 0-based
        sz = v.size();
    }
public:
    // Default: Max heap
    PriorityQueue()
    {
        sz = 0;
        arr.clear();
    }
    PriorityQueue(bool (*cmpr)(const T& t1, const T &t2)) : PriorityQueue()
    {
        this->cmpr = cmpr;
    }
    PriorityQueue(vector<T> v, bool (*cmpr)(const T& t1, const T &t2) = [](const T& t1, const T &t2){return t1 > t2;}) : PriorityQueue()
    {
        sizeInit(v);
        this->cmpr = cmpr;
        construct_heap();
    }
    void construct_heap();
    void push(T t);
    T extract();
    void bst_print();
    int size(){return sz;}
    T top(){return arr[1];}
    bool empty(){return (sz < 1);}
};

template<class T>
void PriorityQueue<T>::heapify(int i) {
    int l = left(i);
    int r = right(i);
    int largest = i;
    if(l <= sz && cmpr(arr[l], arr[i]))
        largest = l;
    if(r <= sz && cmpr(arr[r], arr[largest]))
        largest = r;
    if(i != largest) {
        swap(arr[largest], arr[i]);
        heapify(largest);
    }
}

// waiting for check
template<class T>
void PriorityQueue<T>::push(T t) {
    sz++;
    if(sz >= arr.size())
        arr.resize(sz+1);
    int cur = sz;
    arr[cur] = t;
    while(cur >= 1 && parent(cur) && cmpr(arr[cur], arr[parent(cur)])){
        int p = parent(cur);
        swap(arr[cur], arr[p]);
        cur = p;
    }

    // if(debug)
    // {
    //     cout << "arr: ";
    //     for(auto it: arr)
    //         cout << it << ", ";
    //     cout << endl;
    // }
}

template<class T>
T PriorityQueue<T>::extract() {
    if(sz > 0) {
        T best = arr[1];
        swap(arr[1], arr[sz]);
        sz--;
        heapify(1);
        return best;
    }
    else {
    //    printf("error extracting.\n");
        return T();
    }
}

template<class T>
void PriorityQueue<T>::construct_heap() {
    for(int i = sz / 2; i >= 1; i--)
    {
        heapify(i);
    }
}

template<class T>
void PriorityQueue<T>::bst_print() {
    deque<int> q;
    if(sz < 1)
        return;
    q.push_front(1);
    int s = 1;
    while(q.size())
    {
        int next_s = 0;
        for(int ct = 0; ct < s; ct++)
        {
            int i = q.back();
            q.pop_back();
            if(left(i) <= sz) {
                q.push_front(left(i));
                next_s++;
            }
            if(right(i) <= sz) {
                q.push_front(right(i));
                next_s++;
            }
            cout << arr[i] << ", ";
        }
        s = next_s;
        cout << endl;
    }
}
#endif //PRIORITYQUEUE_H
