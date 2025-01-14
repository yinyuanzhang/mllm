#!/bin/bash
mkdir ../build
cd ../build || exit

cmake .. -DCMAKE_BUILD_TYPE=Debug

make -j$(nproc)
