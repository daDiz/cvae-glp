#!/bin/bash



for ts in 2 4 8 16 32
do
    python preprocess.py -ts $ts > preprocess.log
    for hd in 128 256 512
    do
        for lr in 1.0
        do
            echo "time step $ts h dim $hd lr $lr"
            python train.py -ts $ts -hd $hd -lr $lr > train.log
            python predict_valid.py -ts $ts -hd $hd -lr $lr > predict_valid.log
            python predict_test.py -ts $ts -hd $hd -lr $lr > predict_test.log
        done
    done
done
