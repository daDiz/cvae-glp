#!/bin/bash



for n in 16 32 64
do
    echo "ncp$n"
    python do_svd.py -ncp $n > do_svd.log
    python predict_hw.py -s valid -ncp $n > predict_valid.log
    python predict_hw.py -s test -ncp $n > predict_test.log
done
