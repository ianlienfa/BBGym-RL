# gcc flags
CPP = g++
CPPFLAGS = -std=c++17 -Wall -Wextra

# pybind 
PYBIND = false
PYBIND_EXTRA_CPPFLAGS = -arch arm64e -O3
MAC_SPECIFIC = -undefined dynamic_lookup
PYBIND_LINKER_FLAGS = $(PYBIND_EXTRA_CPPFLAGS) $(MAC_SPECIFIC)
PYBIND_INCLUDE_PATH := $(shell (python3 -m pybind11 --includes))
PYBIND_EXTENSION := $(shell (python3-config --extension-suffix))
PACKAGE_NAME = BB
USER_DEF_PATH = include/user_def
PYBIND_SRC = $(USER_DEF_PATH)/oneRjSumCj_engine.cpp $(USER_DEF_PATH)/oneRjSumCjBranch.cpp $(USER_DEF_PATH)/oneRjSumCjNode.cpp $(USER_DEF_PATH)/oneRjSumCjPrune.cpp  $(USER_DEF_PATH)/oneRjSumCjSearch.cpp 
SCRIPT_PATH = python

# executable 
BIN_NAME = main

# libaraies flags
LIBS = 

# directory flags
INCLUDES = -I include/ $(PYBIND_INCLUDE_PATH)

# path
SRC_PATH = src
BUILD_PATH = build
OBJ_PATH = $(BUILD_PATH)/obj
BIN_PATH = $(BUILD_PATH)/bin
SRC_EXT = cpp
CPLUS_INCLUDE_PATH= include
export CPLUS_INCLUDE_PATH
# REMOVE_BIN=$(shell ls $(BIN_PATH))
# REMOVE_OBJ=$(shell echo "`pwd`"/"`ls $(OBJ_PATH)`")

# code lists
SOURCES = $(shell find src -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
INCLUDE_SOURCES := $(shell find include -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
INCLUDE_SOURCES_MOVE := $(foreach F,$(shell find include -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-),$(lastword $(subst /, ,$F)))
INCLUDE_OBJECTS = $(INCLUDE_SOURCES_MOVE:%.$(SRC_EXT)=build/obj/%.o)
PYBIND_INCLUDE_OBJECTS = $(INCLUDE_SOURCES_MOVE:%.$(SRC_EXT)=build/pybindObj/%.o)
OBJECTS = $(SOURCES:src/%.$(SRC_EXT)=build/obj/%.o) 
vpath %.$(SRC_EXT) include/BB_engine:include/branch_modules:include/problem_parser:include/prune_modules:include/search_modules:include/util:include/user_def


all: $(BIN_PATH)/$(BIN_NAME)
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS) $(INCLUDE_OBJECTS)
	@echo "[Makefile] Linking:   $^ -> $@"	
	$(CPP) $(OBJECTS) $(INCLUDE_OBJECTS) -o $@
	@echo

$(OBJECTS): build/obj/%.o: src/%.$(SRC_EXT)
	@echo "[Makefile] Compiling: $< -> $@"
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
	@echo

# vpath %.$(SRC_EXT) include/BB_engine:include/branch_modules:include/problem_parser:include/prune_modules:include/search_modules:include/util:include/user_def
$(INCLUDE_OBJECTS): build/obj/%.o: %.$(SRC_EXT) 
	@echo "[Makefile] Compiling: $< -> $@"	
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
	@echo

pybind: $(SCRIPT_PATH)/$(PACKAGE_NAME)$(PYBIND_EXTENSION)
$(SCRIPT_PATH)/$(PACKAGE_NAME)$(PYBIND_EXTENSION): $(PYBIND_INCLUDE_OBJECTS)
	@echo "[Makefile] Wrapping: $^ -> $@"	
	$(CPP) $(PYBIND_LINKER_FLAGS)$(CPPFLAGS) $(PYBIND_EXTRA_CPPFLAGS) $(PYBIND_INCLUDE_OBJECTS) -o $@
	@echo

$(PYBIND_INCLUDE_OBJECTS): build/pybindObj/%.o: %.$(SRC_EXT) 
	@echo "[Makefile] Compiling: $< -> $@"	
	$(CPP) $(CPPFLAGS) $(PYBIND_EXTRA_CPPFLAGS) $(PYBIND_INCLUDE_PATH) $(INCLUDES) -c $< -o $@
	@echo


.PHONY: run
run:
	@echo "[Makefile] Running: $(BIN_NAME)"
	@echo "----------------------------------------"
	./$(BIN_PATH)/$(BIN_NAME)
	@echo

.PHONY: debug
debug:
	@echo "[Makefile] Debugging: $(BIN_NAME)"
	@echo "----------------------------------------"
	gdb ./$(BIN_PATH)/$(BIN_NAME)
	@echo

.PHONY: test
test:
	@echo $(SOURCES)
	@echo $(INCLUDE_SOURCES)
	@echo $(OBJECTS)
	@echo $(INCLUDE_OBJECTS)
	
.PHONY: clean
clean:
	rm -f $(OBJECTS) $(INCLUDE_OBJECTS)
	rm -f $(BIN_PATH)/$(BIN_NAME)
	rm -f $(SCRIPT_PATH)/$(PACKAGE_NAME)$(PYBIND_EXTENSION)
	rm -f $(PYBIND_INCLUDE_OBJECTS)
	@echo "[Makefile] removed: $(BIN_PATH)/$(BIN_NAME) $(OBJECTS) $(INCLUDE_OBJECTS)"

.PHONY: up
up:
	rm -f $(OBJECTS) $(INCLUDE_OBJECTS)
	rm -f $(BIN_PATH)/$(BIN_NAME)
	@echo "[Makefile] removed: $(BIN_PATH)/$(BIN_NAME) $(OBJECTS) $(INCLUDE_OBJECTS)"
	make 
