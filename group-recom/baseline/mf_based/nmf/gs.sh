#!/bin/bash



for n in 16 32 64
do
    for l in 0.1 0.5 0.9
    do
    	for a in 0.0 0.1 1.0 10.0
        do
            echo "ncp$n alpha$a l1$l"
            python do_nmf.py -ncp $n -alpha $a -l1 $l > do_nmf.log
	    python predict_hw.py -s valid -ncp $n -alpha $a -l1 $l > predict_valid.log
	    python predict_hw.py -s test -ncp $n -alpha $a -l1 $l > predict_test.log
        done
    done
done
