#!/bin/bash



for n in 32 64 128 256 512 1024
do
    echo "ncp$n"
    python do_svd.py -ncp $n > do_svd.log
    python predict_H.py -s valid -m cos > ./results/H/hit_n"$n"_valid_cos.txt
    python predict_H.py -s test -m cos > ./results/H/hit_n"$n"_test_cos.txt
done
