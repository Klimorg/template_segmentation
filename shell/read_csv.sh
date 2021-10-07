#!/bin/bash

read_csv(){
    INPUT=$1
    OLDIFS=$IFS
    IFS=','
    [ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
    while read filename mask
    do
        echo "filename : $filename"
        echo "mask : $mask"
    done < $INPUT
    IFS=$OLDIFS
}


read_csv $1