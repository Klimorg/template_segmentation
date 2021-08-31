#!/bin/bash

create_dataset(){
    #creation d'un repertoire
    mkdir ./datas/raw_dataset
    #boucle sur tous les éléments d'un repertoire donné
    for f in ./datas/raw_datas/*; do
        if [ -d "$f" ]; then
            # $f is a directory
            b=$(basename $f)
            # on récupère le nom du repertoire
            echo "Making new directories for" $b
            mkdir ./datas/raw_dataset/$b
            #on crée un dossier avec le même nom
            #ls $f/ | head -$1
            echo "Copying the first $1 pictures for folder $b"
            for F in $(ls $f/ | sort | head -$1); do
                cp $f/$F ./datas/raw_dataset/$b/$F
            done
        fi
    done
    echo "Done."
}

create_dataset $1
#mv `ls Positive/ | head -10` ../small_dataset/Positive/
