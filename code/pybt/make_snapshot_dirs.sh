#! /bin/bash

#mkdir snapshots
#cd snapshots
#mkdir 0
#mkdir 1
#mkdir 2
ROOT=/local/disk1/data2/snapshots 

mkdir $ROOT

for dir in 0 1 2;
do
    mkdir $ROOT/$dir
    for symbol in ES NQ GE 6A 6B 6C 6E 6S 6J 6M CL NG ZN ZF ZB;
    do
        mkdir $ROOT/$dir/$symbol
    done
done

