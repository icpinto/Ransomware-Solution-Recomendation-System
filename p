#! /bin/bash
    for f in ~/Desktop/proj/zip/puq/*
        do
        #echo $f
	    #mv $f ~/all_fam
	    #tr '\n' ' ' < $f > ~/Desktop/proj/zip/puq/`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32`.csv
	    echo "" >> $f
    done
