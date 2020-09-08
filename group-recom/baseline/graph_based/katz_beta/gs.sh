#!/bin/bash



for l in 1 2 3 4 5
do
    for b in 0.1 0.01 0.001 0.0001
    do
        for m in sum max min mean
        do
            echo "max l $l beta $b method $m"
            python katz_group.py -f valid -l $l -b $b -m $m > katz_group.log
        done
    done
done
