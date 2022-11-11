#!/bin/bash

export DIR=$(pwd)

export RISCV=/usr/local/research/projects/SystemDesign/tools/riscv/20220414/rv32gcp
export LLVM=/usr/local/research/projects/SystemDesign/tools/llvm/13.0.1/
export SPIKE=/usr/local/research/projects/SystemDesign/tools/spike/20220414/rv32gc/spike
export SPIKEPK=/usr/local/research/projects/SystemDesign/tools/spike/20220414/rv32gc/pk

export PATH=$RISCV/bin/:$LLVM/bin/:$PATH
export PYTHONPATH=$DIR/python/

export VENV=$DIR/venv

test -d $VENV || virtualenv -p python3 $VENV

source $VENV/bin/activate
