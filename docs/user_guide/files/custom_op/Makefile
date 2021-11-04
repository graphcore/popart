# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
CXX ?= g++
CXXFLAGS = -std=c++11 -O3 -shared -fPIC
LDLIBS = -lpopart -lpoplar -lpopops
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = build
SOURCES = custom_op.cpp
TARGET = $(BUILD_DIR)/custom_op.so

all: create_build_dir custom_op 

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

custom_op: custom_op.cpp
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
