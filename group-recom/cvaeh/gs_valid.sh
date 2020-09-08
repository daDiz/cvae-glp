#!/bin/bash


for ts in 2 4 8
do
	for zd in 32 64 128 256
	do
    		for hd in 128 256 512 1024
    		do
        		for lr in 0.0001 0.001 0.01
        		do
            			echo "ts $ts z dim $zd h dim $hd lr $lr"
            			python valid.py -ts $ts -zd $zd -hd $hd -lr $lr > valid.log
        		done
    		done
	done
done
