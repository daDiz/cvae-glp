#!/bin/bash



for d in 32 64 128
do
    for p in 0.5 1.0 1.5
    do
        for q in 0.5 1.0 1.5
        do
            for m in sum max min
            do
                echo "d $d p $p q $q m $m"
                python n2v.py -f valid -d $d -p $p -q $q -m $m
            done
        done
    done
done
