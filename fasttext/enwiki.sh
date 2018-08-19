#!/bin/bash

DATA_ROOT=data/enwiki

function load_data(){
    wget http://mattmahoney.net/dc/enwik9.zip -O $DATA_ROOT/enwik9.zip
    cd $DATA_ROOT && unzip enwik9.zip 
    perl ~/Bigdata/fastText-0.1.0/wikifil.pl enwik9 > fil9
    # head -n 12404 cooking.stackexchange.txt > cooking.train 
    # tail -n 3000 cooking.stackexchange.txt > cooking.valid
}

function skipgram(){
    fasttext skipgram -input $DATA_ROOT/fil9 -output output/fil9
}

# load_data

skipgram

# fasttext nn output/fil9.bin
