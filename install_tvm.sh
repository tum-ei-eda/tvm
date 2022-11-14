#!/bin/bash

# exit when any command fails
set -e

if [ -z ${DIR} ]
then
    echo "Please run \`source env.sh\` first!"
    exit 1
fi

git submodule update --init --recursive

echo "Installing python packages..."
pip install -r $DIR/requirements.txt

mkdir -p $DIR/build

test -f $DIR/build/config.cmake || cp $DIR/cmake/config.cmake $DIR/build/config.cmake

echo "Configuring TVM..."
sed -i "s|set(USE_MICRO OFF)|set(USE_MICRO ON)|g" $DIR/build/config.cmake
sed -i "s|set(USE_MICRO_STANDALONE_RUNTIME OFF)|set(USE_MICRO_STANDALONE_RUNTIME ON)|g" $DIR/build/config.cmake
sed -i "s|set(USE_LLVM OFF)|set(USE_LLVM ON)|g" $DIR/build/config.cmake

cmake -B $DIR/build $DIR

echo "Compiling TVM (this will take a while)"
make -C $DIR/build -j$(nproc)

echo "Done!"
