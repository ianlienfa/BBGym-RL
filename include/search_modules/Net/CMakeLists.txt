add_library(
    NetDDPRActor
    DDPR/NetDDPRActor.h
    DDPR/NetDDPRActor.cpp
)
add_library(
    NetDDPRQNet
    DDPR/NetDDPRQNet.h
    DDPR/NetDDPRQNet.cpp
)
add_library(
    NetDDPR
    DDPR/NetDDPR.h
    DDPR/NetDDPR.cpp
)
add_library(
    DDPR
    DDPR/DDPR.h
)

# PPO
add_library(
    NetPPOActor
    PPO/NetPPOActor.h
    PPO/NetPPOActor.cpp
)
add_library(
    NetPPOQNet
    PPO/NetPPOQNet.h
    PPO/NetPPOQNet.cpp
)
add_library(
    NetPPO
    PPO/NetPPO.h
    PPO/NetPPO.cpp
)
add_library(
    PPO
    PPO/PPO.h
)
# target_include_directories(DDPR INTERFACE ./DDPR)

target_link_libraries(DDPR NetDDPR oneRjSumCjGraph)
target_link_libraries(PPO NetPPO oneRjSumCjGraph)

