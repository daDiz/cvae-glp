#!/bin/bash

for ts in 2 4 8 16 32
do
    python preprocess.py $ts
    for zd in 32 64 128 256
    do
        for hd in 128 256 512 1024
        do
            for lr in 0.00001 0.0001 0.001 0.01
            do
                echo "ts $ts zdim $zd hdim $hd lr $lr"
                python train.py -ts $ts -zd $zd -hd $hd -lr $lr > train.log
            done
        done
    done
done
