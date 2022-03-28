
add_library(
    problemNode
    BB_engine/problemNode.h
    BB_engine/problemNode.cpp
)

add_library(
    searchGraph
    BB_engine/searchGraph.h
    BB_engine/searchGraph.cpp
)

add_library(
    branchMod
    branch_modules/branchMod.h
    branch_modules/branchMod.cpp
)

add_library(
    pruneMod
    prune_modules/pruneMod.h
    prune_modules/pruneMod.cpp
)

add_library(
    searchMod
    search_modules/searchMod.h
    search_modules/searchMod.cpp
)

add_library(
    config
    util/config.h
    util/config.cpp
)

add_library(
    problemParser
    problem_parser/problemParser.h
    problem_parser/problemParser.cpp
)

add_library(
    LowerBound
    util/LowerBound.h
    util/LowerBound.cpp
)

add_library(
    PriorityQueue
    util/PriorityQueue.h
    util/PriorityQueue.cpp
)

add_library(
    types
    util/types.h
    util/types.cpp
)

add_library(
    JobWithReleaseDate
    util/JobWithReleaseDate.h
    util/JobWithReleaseDate.cpp
)

add_library(
    TorchUtil
    INTERFACE 
)
target_include_directories(TorchUtil INTERFACE util)

add_subdirectory(user_def)
add_subdirectory(search_modules)
add_subdirectory(third_party)
