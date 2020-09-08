#!/bin/bash



for hd in 64 128 256 512 1024
do
    for lr in 0.0001 0.001 0.01 0.1
    do
    	echo "hd$hd lr$lr"
        python train.py -hd $hd -lr $lr > train.log
    done
done
