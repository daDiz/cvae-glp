#!/bin/bash


for ed in 32 64 128 256
do 
    for hd in 128 256 512 1024
    do
        for lr in 0.001 0.01 0.1
        do
	    for ts in 1
	    do
    		echo "ed$ed hd$hd lr$lr ts$ts"
        	python valid.py -ed $ed -hd $hd -lr $lr -ts $ts > valid.log
    	    done
	done
    done
done
