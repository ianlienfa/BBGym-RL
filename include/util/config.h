#ifndef CONFIG_H
#define CONFIG_H

#define JOB_NUMBER 30
#define TIME_TYPE float

// exit codes
#define NOT_INIT 50
#define INVALID_INPUT 51
#define LOGIC_ERROR 52

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
#define bundle_OneRjSumCj_LU_AND_SAL 1
#define bundle_OneRjSumCj_CBFS 2

/* Simple Strategy Choices */
#define SEARCH_BUNDLE bundle_OneRjSumCj_CBFS

#if SEARCH_BUNDLE == bundle_OneRjSumCj_LU_AND_SAL
    #define SEARCH_STRATEGY 1
    #define BRANCH_STRATEGY 1
    #define PRUNE_STRATEGY  1
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_unify
#endif

#if SEARCH_BUNDLE == bundle_OneRjSumCj_CBFS
    #define SEARCH_STRATEGY 0
    #define BRANCH_STRATEGY 1
    #define PRUNE_STRATEGY  0
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_unify
#endif

/* Problem specific specifications <-- Change Specific Solving Strategies Here!
   Note that some of the strategies are not applicable to all problems.
 */
#if SEARCH_BUNDLE == 0 
    #define SEARCH_STRATEGY 1
    #define BRANCH_STRATEGY 1
    #define PRUNE_STRATEGY  1
    #define LOWER_BOUND lowerbound_oneRjSumCj_LU_AND_SAL
    #define LABELER labeler_unify
#endif 
/* PRUNE STRATEGY CAN BE CHOOSED BY PLACING FUNCTION IN OneRjSUMCjPrune::prune_funcs */

/* Tracing Functions */
#ifndef PRE_SOLVE_PRINT_CONFIG
    #define PRE_SOLVE_PRINT_CONFIG()  print_config()
#endif 

#ifndef POST_SOLVE_PRINT_CONFIG
    #define POST_SOLVE_PRINT_CONFIG(graph)  post_print_config(graph)
#endif 

/* debugging */
#define DEBUG_LEVEL 2

/* validations (some extra checking is done if validation level is high) */
#define VALIDATION_LEVEL validation_level_HIGH

#endif