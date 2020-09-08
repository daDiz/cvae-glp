#!/bin/bash



for ts in 5 10 15
do
    for hd in 128 256 512
    do
        for lr in 0.0001 0.001 0.01
        do
            echo "time step $ts h dim $hd lr $lr"
            python train.py -ts $ts -hd $hd -lr $lr > train.log
        done
    done
done
