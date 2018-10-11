#!/bin/bash

mkdir build
cd build
cmake -DEIGEN_INCLUDE_DIR="./../third_party/include/eigen3"
mkdir ./bin/result
make
