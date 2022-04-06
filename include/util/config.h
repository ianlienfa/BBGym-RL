#ifndef CONFIG_H
#define CONFIG_H

#define JOB_NUMBER 100
#define TIME_TYPE float
#define FIXED_JOB_SIZE 100
#define CONTOUR_TYPE float

// exit codes
#define NOT_INIT 50
#define INVALID_INPUT 51
#define LOGIC_ERROR 52

// restarts modela
#define QNetPath "../saved_model/qNet.pt"
#define PiNetPath "../saved_model/piNet.pt"
#define PiOptimPath "../saved_model/optimizer_pi.pt"
#define QOptimPath "../saved_model/optimizer_q.pt"
#define INF_MODE 0


/* Available Search Options Definitions */
#define searchOneRjSumCj_CBFS 0
#define searchOneRjSumCj_LU_AND_SAL 1

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

/* ==================== Simple Strategy Choices ========================== */

/* Simple Strategy Choices */
#define SEARCH_BUNDLE bundle_OneRjSumCj_CBFS_Net


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
    #define POST_SOLVE_PRINT_CONFIG(graph)  post_print_config(graph)
#endif 

#ifndef SOLVE_CALLBACK    
    extern void solveCallbackImpl( void* );
    #define SOLVE_CALLBACK(engine) solveCallbackImpl(engine)
#endif

#ifndef OPTIMAL_FOUND_CALLBACK    
    extern void optimalFoundCallbackImpl( void* );
    #define OPTIMAL_FOUND_CALLBACK(engine) optimalFoundCallbackImpl(engine)
#endif

/* debugging */
#define DEBUG_LEVEL 0
#define TORCH_DEBUG 0

/* validations (some extra checking is done if validation level is high) */
#define VALIDATION_LEVEL validation_level_HIGH

#endif