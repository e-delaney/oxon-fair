#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES

export usermode="data=jigsawreligion,mfair=[christian,other,muslim]"
python $DBG bert.py