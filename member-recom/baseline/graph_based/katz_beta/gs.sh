#!/bin/bash



for l in 1 2 3 4 5
do
    for b in 0.1 0.01 0.001 0.0001
    do
        echo "max l $l beta $b"
        python katz.py -f valid -l $l -b $b
    done
done
