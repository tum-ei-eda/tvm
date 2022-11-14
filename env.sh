#!/bin/bash

# exit when any command fails
# set -e

DOWNLOAD=1

export DIR=$(pwd)

if [[ "$DOWNLOAD" == 1 ]]
then
    # Initialize download dir
    export DL_DIR=$DIR/dl
    mkdir -p $DL_DIR

    echo "Downloading dependencies to $DL_DIR"

    # Download RISC-V toolchain
    export RISCV=$DL_DIR/rv32gcp
    if [[ ! -d $RISCV ]]
    then
        wget https://syncandshare.lrz.de/dl/fiNvP4mzVQ8uDvgT9Yf2bqNk/rv32gcp.tar.xz -O $DL_DIR/rv32gcp.tar.xz
        mkdir -p $RISCV
        tar -xf $DL_DIR/rv32gcp.tar.xz -C $RISCV
        rm $DL_DIR/rv32gcp.tar.xz
        echo "Successfully installed RISC-V toolchain (RV32GCP)."
    else
        echo "RISC-V toolchain (RV32GCP) is already installed."
    fi

    # Download LLVM
    export LLVM=$DL_DIR/llvm13
    if [[ ! -d $LLVM ]]
    then
        wget https://syncandshare.lrz.de/dl/fiDmr2yoqLFDokH3DjT9eJ/llvm13.tar.gz -O $DL_DIR/llvm13.tar.gz
        mkdir -p $LLVM
        tar -xf $DL_DIR/llvm13.tar.gz -C $LLVM
        rm $DL_DIR/llvm13.tar.gz
        echo "Successfully installed LLVM."
    else
        echo "LLVM is already installed.."
    fi

    # Download Spike Simulator
    export SPIKE=$DL_DIR/riscv-isa-sim/spike
    export SPIKEPK=$DL_DIR/riscv-isa-sim/pk
    if [[ ! -f $SPIKE ]] || [[ ! -f $SPIKEPK ]]
    then
        wget https://syncandshare.lrz.de/dl/fiUzs9EAkeYKDrvW1pjrBt/spike_rv32gc.tar.gz -O $DL_DIR/riscv-isa-sim.tar.gz
        mkdir -p $DL_DIR/riscv-isa-sim/
        tar -xf $DL_DIR/riscv-isa-sim.tar.gz -C $DL_DIR/riscv-isa-sim --strip-components=1
        rm $DL_DIR/riscv-isa-sim.tar.gz
        echo "Successfully installed riscv-isa-sim."
    else
        echo "riscv-isa-sim is already installed.."
    fi

else
    export RISCV=/usr/local/research/projects/SystemDesign/tools/riscv/20220414/rv32gcp
    export LLVM=/usr/local/research/projects/SystemDesign/tools/llvm/13.0.1/
    export SPIKE=/usr/local/research/projects/SystemDesign/tools/spike/20220414/rv32gc/spike
    export SPIKEPK=/usr/local/research/projects/SystemDesign/tools/spike/20220414/rv32gc/pk
fi

export PATH=$RISCV/bin/:$LLVM/bin/:$PATH
export PYTHONPATH=$DIR/python/

export VENV=$DIR/venv

test -d $VENV || virtualenv -p python3 $VENV

echo "Entering virtual python environment"
source $VENV/bin/activate

echo "Initialized shell environment!"
