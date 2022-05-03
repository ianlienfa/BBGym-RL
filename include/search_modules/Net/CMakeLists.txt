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
# target_include_directories(DDPR INTERFACE ./DDPR)

target_link_libraries(DDPR NetDDPR oneRjSumCjGraph)

