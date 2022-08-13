
#ifndef CONFIG_H
#define CONFIG_H

#define JOB_NUMBER 100
#define TIME_TYPE float
#define FIXED_JOB_SIZE 100
#define CONTOUR_TYPE float
#define RANDOM_SEED 50
const float node_reward = -1e-2;
const float move_reward = -1e-4;
const float move_encouragement_reward = 1e-13;
const float end_emphasize_multiplier = 1.5;
const float move_reward_min = -1e-2;
const float move_reward_slope = 3;
const float neg_zero_reward = -1e-7;
const float pos_zero_reward = 1e-7;

// Grid search parameters
#ifndef V_MAX_NUM_CNTR
    #define V_MAX_NUM_CNTR 10
#endif
#ifndef V_LR_PI
    #define V_LR_PI 3e-6
#endif
#ifndef V_LR_Q
    #define V_LR_Q 3e-5
#endif

// #define NDEBUG

// exit codes
#define NOT_INIT 50
#define INVALID_INPUT 51
#define LOGIC_ERROR 52

// restarts modela
#define QNetPath "../saved_model/qNet.pt"
#define PiNetPath "../saved_model/piNet.pt"
#define PiOptimPath "../saved_model/optimizer_pi.pt"
#define QOptimPath "../saved_model/optimizer_q.pt"
#define QNetPathInf "../saved_model/inf/qNet.pt"
#define PiNetPathInf "../saved_model/inf/piNet.pt"
#define PiOptimPathInf "../saved_model/inf/optimizer_pi.pt"
#define QOptimPathInf "../saved_model/inf/optimizer_q.pt"
#define PlainCBFSVerbose "base-compute" 
#define PlainCBFSBFS "base-compute-pure"
#define PlainCBFSLevel "base-compute-level"
#define PlainCBFSRand "base-compute-rand"
#define INF_MODE 0
#define MEASURE_MODE 0
#define LAYER_WEIGHT_PRINT 1
// #define NDEBUG 


/* Available Search Options Definitions */
#define searchOneRjSumCj_CBFS 0
#define searchOneRjSumCj_LU_AND_SAL 1
#define searchOneRjSumCj_CBFS_LIST 2

/* Available Branch Options Definitions */
#define branchOneRjSumCj_PLAIN 0
#define branchOneRjSumCj_LU_AND_SAL 1

/* Available Prune Options Definitions */
#define pruneOneRjSumCj_plain 0
#define pruneOneRjSumCj_LU_AND_SAL 1

/* Available Validation level Definitions */
#define validation_level_NONE 0
#define validation_level_HIGH 5

/* Available Lowerbound Definitions */
// ----- 1|rj|Sum_Cj ----- //
#define lowerbound_oneRjSumCj_LU_AND_SAL 0

/* Available Labeler Definitions */
#define labeler_unify 0
#define labeler_bylevel 1
#define labeler_bynet 2

/* Available Solver Bundles */
#define bundle_NULL 0
#define bundle_OneRjSumCj_LU_AND_SAL 1
#define bundle_OneRjSumCj_CBFS 2
#define bundle_OneRjSumCj_CBFS_Net 3
#define bundle_OneRjSumCj_CBFS_LIST 4

/* ==================== Simple Strategy Choices ========================== */

/* Simple Strategy Choices */
#define SEARCH_BUNDLE bundle_OneRjSumCj_CBFS_LIST


/* ======================================================================= */
#if SEARCH_BUNDLE == bundle_OneRjSumCj_LU_AND_SAL
    #define SEARCH_STRATEGY 1
    #define BRANCH_STRATEGY 1
    #define PRUNE_STRATEGY  1
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_unify
#endif

#if SEARCH_BUNDLE == bundle_OneRjSumCj_CBFS
    #define SEARCH_STRATEGY searchOneRjSumCj_CBFS
    #define BRANCH_STRATEGY branchOneRjSumCj_LU_AND_SAL
    #define PRUNE_STRATEGY  pruneOneRjSumCj_plain
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_unify
#endif

#if SEARCH_BUNDLE == bundle_OneRjSumCj_CBFS_Net
    #define SEARCH_STRATEGY searchOneRjSumCj_CBFS
    #define BRANCH_STRATEGY branchOneRjSumCj_LU_AND_SAL
    #define PRUNE_STRATEGY  pruneOneRjSumCj_plain
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_bynet
#endif

#if SEARCH_BUNDLE == bundle_OneRjSumCj_CBFS_LIST
    #define SEARCH_STRATEGY searchOneRjSumCj_CBFS_LIST
    #define BRANCH_STRATEGY branchOneRjSumCj_LU_AND_SAL
    #define PRUNE_STRATEGY  pruneOneRjSumCj_plain
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_bynet
#endif

/* Problem specific specifications <-- Change Specific Solving Strategies Here!
   Note that some of the strategies are not applicable to all problems.
 */
#if SEARCH_BUNDLE == bundle_NULL 
    #define SEARCH_STRATEGY 
    #define BRANCH_STRATEGY 
    #define PRUNE_STRATEGY  
    #define LOWER_BOUND 
    #define LABELER 
#endif 
/* PRUNE STRATEGY CAN BE CHOOSED BY PLACING FUNCTION IN OneRjSUMCjPrune::prune_funcs */

/* Tracing Functions */
#ifndef PRE_SOLVE_PRINT_CONFIG
    #define PRE_SOLVE_PRINT_CONFIG()  print_config()
#endif 

#ifndef POST_SOLVE_PRINT_CONFIG
    #define POST_SOLVE_PRINT_CONFIG(engine)  post_print_config(engine)
#endif 

#ifndef SOLVE_CALLBACK    
    extern void solveCallbackImpl( void* );
    #define SOLVE_CALLBACK(engine) solveCallbackImpl(engine)
#endif

#ifndef OPTIMAL_FOUND_CALLBACK    
    extern void optimalFoundCallbackImpl( void* );
    #define OPTIMAL_FOUND_CALLBACK(engine) optimalFoundCallbackImpl(engine)
#endif

#ifndef EARLY_STOPPING_CALLBACK    
    extern void earlyStoppingCallbackImpl( void* );
    #define EARLY_STOPPING_CALLBACK(engine) earlyStoppingCallbackImpl(engine)
#endif

/* debugging */
#define DEBUG_LEVEL 0
#define TORCH_DEBUG -2

/* validations (some extra checking is done if validation level is high) */
#define VALIDATION_LEVEL validation_level_HIGH

#endif