#include <iostream>
#include <iomanip>
#include <list>
#include <limits>
#include <cmath>
#include "util/types.h"
#include "util/config.h"
#include "util/PriorityQueue.h"
using std::cout, std::endl;
using std::list;
using std::numeric_limits;

template<typename T, typename Len>
struct PlacementList{
private:
    Len max_size = numeric_limits<Len>::max();
    bool (*cmpr)(const T &i1, const T &i2) = nullptr;
public:
    Len current_pos;  // now searched position
    Len picker_pos;   // picker position for placement
    typename list<PriorityQueue<T>>::iterator current_iter;
    typename list<PriorityQueue<T>>::iterator picker_iter;
    list<PriorityQueue<T>> lst;    
    int picker_steps = 0; // picker step counter trackes the number of steps the picker has taken before a placement

    PlacementList(){
    }
    void init(bool(*cmp)(const T &i1, const T &i2))        
    {
        cmpr = cmp;
        lst.push_back(PriorityQueue<T>(cmpr));
        current_iter = lst.begin();     
        picker_iter = lst.begin();   
        current_pos = 0;
        picker_pos = 0;
    }
    
    void picker_step_reset()
    {
        picker_steps = 0;
        picker_iter = current_iter;
        picker_pos = current_pos;
    } 

    float get_picker_reward()
    {
        const auto& move_reward_base = move_reward;
        return (picker_steps < 10) ? move_encouragement_reward : std::max(move_reward_base * float(pow(2.0, picker_steps - 10)), move_reward_min); // increase negative reward exponentially
    }

    Len size() const{
        int64_t pos = 0;
        // do some synchronization check, not related to size
        for(typename list<PriorityQueue<T>>::const_iterator iter = lst.cbegin(); iter != lst.cend(); iter++){
            if(iter != current_iter)         
            pos++;
            else{
            if(current_pos != pos)
            {
                print();
                cout << "current_pos: " << current_pos << " pos: " << pos << endl;                
            }
            assertm("position not synced", current_pos == pos);
            break;
            }
        }         
        return lst.size();
    } 
    bool max_size_set() const {
        return max_size != numeric_limits<Len>::max();
    }
    void set_max_size(Len max_size){this->max_size = max_size;}    
    bool empty() const {return lst.empty();}
    typename list<PriorityQueue<T>>::iterator get_current_iter(){return current_iter;}
    Len left() {   
        GAME_TRACK("left",
        picker_steps++;
        if(picker_iter == lst.end() || picker_pos == numeric_limits<Len>::max()) 
        {
            assertm("picker_iter should sync with picker_pos", (picker_iter == lst.end() && picker_pos == numeric_limits<Len>::max()));
            picker_iter = current_iter;
            picker_pos = current_pos;
        }
        if(this->empty())
            assertm("contour is empty, undefined behavior", false);
        if(picker_iter == lst.begin())
        {            
            assertm("position not synced", picker_pos == 0);
            picker_iter = (--lst.end());
            picker_pos = lst.size() - 1;
        }
        else
        {
            picker_iter--;
            picker_pos--;
        }      
        );
        return picker_pos;
    }
    Len right() {  
        GAME_TRACK("right",        
        picker_steps++;
        if(picker_iter == lst.end() || picker_pos == numeric_limits<Len>::max()) 
        {
            assertm("picker_iter should sync with picker_pos", (picker_iter == lst.end() && picker_pos == numeric_limits<Len>::max()));
            picker_iter = current_iter;
            picker_pos = current_pos;
        }
        if(this->empty())
            assertm("contour is empty, undefined behavior", false);
        if(picker_iter == (--lst.end()))
        {          
            assertm("position not synced", picker_pos == lst.size() - 1);
            picker_iter = lst.begin();
            picker_pos = 0;
        }
        else
        {         
            picker_iter++;
            picker_pos++;
        }       
        );       
        return picker_pos;
    }

    Len _place(T element){
        if(cmpr == nullptr)
        {
            assertm("comparator not set, call init() before calling place", false);
        }
        picker_iter->push(element);
        auto placed_pos = picker_pos;
        picker_step_reset();
        return placed_pos;
    }
    
    Len place(T element){
        GAME_TRACK("place",
        Len picker_pos = _place(element);
        );
        return picker_pos;
    }

    Len insert_and_place(T element){
        GAME_TRACK("insert_and_place",
        if(cmpr == nullptr)
        {
            assertm("comparator not set, call init() before calling place", false);
        }
        if(this->empty())
            assertm("contour list is empty, undefined behavior", false);

        // if current contour is empty, directly place it in
        if(picker_iter->empty()) // happens when intended to place in the contour that the parent is poped from
        {
            picker_iter->push(element);            
            return current_pos;
        }

        // Increase contour
        if(lst.size() < max_size){
            if(picker_iter == (--lst.end())){
                cout << "insert: place at end" << endl;
                // check the max size, only increase if it is not reached
                
                lst.push_back(PriorityQueue<T>(cmpr));
                picker_iter = (--lst.end());
                picker_pos = lst.size() - 1;            
            }
            else
            {
                picker_iter++;
                picker_pos++;
                if(current_pos >= picker_pos) // inserted in front of current_iter, current_iter should be pushed right
                {
                    current_pos++;
                }            
                picker_iter = lst.emplace(picker_iter, PriorityQueue<T>(cmpr));                                    
            }
        }
        else
        {
            #if GAME_TRACKER == 1
            cout << "insert: max size reached, inserting into current position" << endl;                
            #endif
        }

        // push element
        #if GAME_TRACKER == 1
        cout << "pushing element: " << element << endl;
        #endif
        auto placed_pos = _place(std::move(element));
        picker_step_reset();
        );
        return placed_pos;
    }

    Len erase_contour()
    {                
        if(!current_iter->empty())
        {            
            cout << "contour empty" << endl;
            assertm("contour is not empty, undefined behavior", false);
        }                
        return erase_contour(current_iter);
    }

    Len erase_contour(typename list<PriorityQueue<T>>::iterator iter)
    {
        GAME_TRACK("erasing_contour", 
        if(this->empty())
            assertm("contour already empty", false);
        current_iter = lst.erase(iter);
        if(current_iter == lst.end())
        {
            current_iter = lst.begin();
            current_pos = 0;
        }
        picker_iter = current_iter;
        picker_pos = current_pos;
        );
        return current_pos;
    }

    Len step_forward()
    {
        GAME_TRACK("step_forward",     
        if(this->empty())
            assertm("contour already empty", false);
        assertm("current_iter should not be at end", current_iter != lst.end());
        current_iter++;
        current_pos++;
        if(current_iter == lst.end())
        {
            current_iter = lst.begin();
            current_pos = 0;
        }
        picker_step_reset();
        );
        return current_pos;
    }

    T pop_from_current()
    {
        T current_element = current_iter->top();
        current_iter->extract();
        return current_element;
    }

    void print() const
    {
        // cout << "contour size: " << this->size() << endl;
        int pos = 0;
        for(typename list<PriorityQueue<T>>::const_iterator iter = lst.cbegin(); iter != lst.cend(); iter++){
            cout << std::setw(3);
            if(iter == picker_iter)
                cout << "> ";
            else
                cout << "  ";
            if(iter == current_iter)
                cout << "*";
            else
                cout << " ";        
                
            printf("[%3d] ", pos);
            iter->bst_print();
            cout << endl;
            pos++;
        }
    }

    vector<float> get_snapshot() const{
        if(max_size == numeric_limits<Len>::max())
        {
            assertm("max_size not set, call set_max_size() before calling get_snapshot()", false);
        }
        const float norm_factor = 1e3;
        vector<float> contour_snapshot;        
        assertm("contour size exceeds max_num_contour", max_size >= lst.size());
        contour_snapshot.assign(max_size, -1e-11);      
        int i = 0;  
        for(const auto &iter: lst){
            if(!iter.empty())
            {
                contour_snapshot[i] = ((float)iter.size()) / norm_factor; 
                assertm("snapshot with zero value", contour_snapshot[i] != 0.0);  
                i++;
            }
        }        
        return contour_snapshot;
    }
};
