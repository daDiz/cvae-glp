#!/bin/bash



for n in 64 128 256 512 1024
do
    for l in 0.1 0.5 0.9
    do
    	for a in 0.0 0.1 1.0 10.0
        do
            echo "ncp$n alpha$a l1$l"
            python do_nmf.py -ncp $n -alpha $a -l1 $l > do_nmf.log
	    python predict_H.py -s valid -m cos > ./results/H/hit_n"$n"_a"$a"_l"$l"_valid_cos.txt
	    python predict_H.py -s test -m cos > ./results/H/hit_n"$n"_a"$a"_l"$l"_test_cos.txt
        done
    done
done
