#!/bin/bash

# GPU
export THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn,warn_float64=warn,lib.cnmem=0.9'

# CPU
# export THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32'

nohup python -u ./evaluate.py model_0 0 > log0.txt 2>&1 &
#python ./train.py



