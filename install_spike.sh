#!/bin/bash

# exit when any command fails
set -e

if [ -z ${DIR} ]
then
    echo "Please run \`source env.sh\` first!"
    exit 1
fi

if ! [ -x "$(command -v dtc)" ]; then
  echo 'Error: dtc is not installed.' >&2
  echo "Please run: 'sudo apt-get install device-tree-compiler'"
  exit 1
fi



mkdir -p $DIR/dl/riscv-isa-sim/
rm -rf $DIR/dl/riscv-isa-sim/*

echo "Cloning Spike..."
test -d $DIR/dl/spike || git clone https://github.com/riscv-software-src/riscv-isa-sim.git $DIR/dl/spike

echo "Compiling Spike..."

mkdir -p $DIR/dl/spike/build
cd $DIR/dl/spike/build
git checkout 0bc176b3fca43560b9e8586cdbc41cfde073e17a
../configure --prefix=$RISCV
make -j$(nproc)
cp spike $DIR/dl/riscv-isa-sim/
cd -


echo "Cloning Proxy Kernel..."
test -d $DIR/dl/spikepk || git clone https://github.com/riscv-software-src/riscv-pk.git $DIR/dl/spikepk

echo "Compiling Proxy Kernel..."

export RISCV=

mkdir -p $DIR/dl/spikepk/build
cd $DIR/dl/spikepk/build
../configure --prefix=$RISCV --host=riscv32-unknown-elf --with-arch=rv32imafdc --with-abi=ilp32d
make -j$(nproc) VERBOSE=1
cp pk $DIR/dl/riscv-isa-sim/
cd -



echo "Done!"
