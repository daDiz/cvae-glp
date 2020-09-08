#!/bin/bash



for zd in 32 64 128 256
do
    for hd in 128 256 512 1024
    do
        for lr in 0.0001 0.001 0.01
        do
            echo "z dim $zd h dim $hd lr $lr"
            python train.py -zd $zd -hd $hd -lr $lr > train.log
        done
    done
done
