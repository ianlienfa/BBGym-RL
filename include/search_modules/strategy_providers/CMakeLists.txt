
add_library(
    Labeler
    INTERFACE 
)
target_include_directories(Labeler INTERFACE .)

add_library(
    PlainLabeler
    PlainLabeler.h
    PlainLabeler.cpp
)

add_library(
    DDPRLabeler
    DDPRLabeler.h
    DDPRLabeler.cpp
)

add_library(
    PPOLabeler
    PPOLabeler.h
    PPOLabeler.cpp
)

target_link_libraries(DDPRLabeler NetDDPR)
target_link_libraries(PPOLabeler NetPPO)