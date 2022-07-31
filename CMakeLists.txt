# general project settings
cmake_minimum_required(VERSION 3.12)
project(BranchAndBoundSmart CXX)


############### DEBUG ###############
set(CMAKE_BUILD_TYPE Debug)


############### GRID SEARCH VARIABLES ###############
add_compile_definitions(V_HIDDEN_DIM=${V_HIDDEN_DIM})
add_compile_definitions(V_MAX_NUM_CNTR=${V_MAX_NUM_CNTR})
add_compile_definitions(V_LR_PI=${V_LR_PI})
add_compile_definitions(V_LR_Q=${V_LR_Q})


############### ARCHITECTURE-RELATED VARIABLES ###############
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()


################ PRE-PACKAGE SEARCH VARIABLES SETUP ################
# general variables
set(CMAKE_OSX_ARCHITECTURES "x86_64") # For m1 mac
set(CMAKE_CXX_STANDARD 20)
set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(HOME_INC_DIR ${PROJECT_PATH}/include)
add_compile_options(-Wall -g)

# Torch related variables
set(TORCH_PREFIX_PATH "./libtorch")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${TORCH_PREFIX_PATH}")

# NumPy related variables
if(MACOSX)
  set(PYTHON_NUMPY_INCLUDE_DIR "/opt/homebrew/Caskroom/miniforge/base/envs/ML/lib/python3.9/site-packages/numpy/core/include")
endif()

# Google Test related variables
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/af29db7ec28d6df1c7f0f745186884091e602e07.zip
# )
# FetchContent_MakeAvailable(googletest)
######################  PACKAGE DEPENDENCIES  ########################
# PACKAGES 
find_package(Torch REQUIRED)  # Torch

# matlablib dependecies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(Eigen3)

if (${EIGEN3_FOUND})
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  message(STATUS "Eigen3 not found")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
if (${Python3_FOUND})
  include_directories(${Python3_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Python3 not found, please install it.")
endif()

find_package(NumPy)
if (${PYTHON_NUMPY_FOUND})
  include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
else()
  message(WARNING "Python3 NumPy not found, proceeding with -DWITHOUT_NUMPY."
  " Some functions might not work.")
  add_definitions(-DWITHOUT_NUMPY)
endif()

################# POST-PACKAGE SEARCH VARIABLES SETUP ###############
# Package specific configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# setting include root path
include_directories(${HOME_INC_DIR})

# subdirectories
add_subdirectory(include)

###########################################################################
# executables

add_executable(main src/main.cpp)
target_link_libraries(
    main    # target 
    "${TORCH_LIBRARIES}" TorchUtil # Torch 
    "${Python3_LIBRARIES}" # Python3
    problemNode searchGraph problemParser searchMod pruneMod branchMod Labeler types PlainLabeler# Base
    oneRjSumCjNode NetDDPRActor NetDDPRQNet NetDDPR DDPRLabeler NetPPOActor NetPPOQNet NetPPO PPOLabeler# Net
    oneRjSumCjGraph oneRjSumCj_engine oneRjSumCjBranch oneRjSumCjPrune oneRjSumCjSearch # OneRjSumCj
    LowerBound PriorityQueue JobWithReleaseDate PlacementList# Util
)

add_executable(net src/net.cpp)
target_link_libraries(
    net    # target 
    "${TORCH_LIBRARIES}" TorchUtil # Torch 
    "${Python3_LIBRARIES}" # Python3
    problemNode searchGraph problemParser searchMod pruneMod branchMod Labeler types PlainLabeler# Base
    oneRjSumCjNode NetDDPRActor NetDDPRQNet NetDDPR DDPRLabeler NetPPOActor NetPPOQNet NetPPO PPOLabeler# Net
    oneRjSumCjGraph oneRjSumCj_engine oneRjSumCjBranch oneRjSumCjPrune oneRjSumCjSearch # OneRjSumCj
    LowerBound PriorityQueue JobWithReleaseDate PlacementList# Util
)

add_executable(load src/load.cpp)
target_link_libraries(
    load    # target 
    "${TORCH_LIBRARIES}" TorchUtil # Torch 
    "${Python3_LIBRARIES}" # Python3
    problemNode searchGraph problemParser searchMod pruneMod branchMod Labeler types PlainLabeler# Base
    oneRjSumCjNode NetDDPRActor NetDDPRQNet NetDDPR DDPRLabeler # Net
    oneRjSumCjGraph oneRjSumCj_engine oneRjSumCjBranch oneRjSumCjPrune oneRjSumCjSearch # OneRjSumCj
    LowerBound PriorityQueue JobWithReleaseDate PlacementList # Util
)

######################  SELF-DEFINED COMMANDS  ########################
# cmake run
add_custom_target(run
    ./main
    DEPENDS main
    WORKING_DIRECTORY ${PROJECT_PATH}/build
)

add_custom_target(cls    
    rm -f ../saved_model/*.pt
    rm -f ../saved_model/*.txt
    WORKING_DIRECTORY ${PROJECT_PATH}/build
)


# enable_testing()
# add_executable(Gtest src/Gtest.cpp)
# target_link_libraries(
#     Gtest    # target 
#     "${TORCH_LIBRARIES}" TorchUtil # Torch 
#     "${Python3_LIBRARIES}" # Python3
#     problemNode searchGraph problemParser searchMod pruneMod branchMod Labeler types PlainLabeler# Base
#     oneRjSumCjNode NetDDPRActor NetDDPRQNet NetDDPR DDPRLabeler # Net
#     oneRjSumCjGraph oneRjSumCj_engine oneRjSumCjBranch oneRjSumCjPrune oneRjSumCjSearch # OneRjSumCj
#     LowerBound PriorityQueue JobWithReleaseDate # Util
#     gtest_main # Google Test
# )
# include(GoogleTest)
# gtest_discover_tests(Gtest)
