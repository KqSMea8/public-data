#!/bin/bash

DATA_ROOT=data/cooking

function load_data(){
    wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cooking.stackexchange.tar.gz -O $DATA_ROOT/cooking.stackexchange.tar.gz
    cd $DATA_ROOT && tar xvzf cooking.stackexchange.tar.gz 
    head -n 12404 cooking.stackexchange.txt > cooking.train 
    tail -n 3000 cooking.stackexchange.txt > cooking.valid
}

function preprocess(){
    PREPROCESS_DATA=$DATA_ROOT/cooking.preprocessed.txt 
    cat $DATA_ROOT/cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > $PREPROCESS_DATA
    head -n 12404 $PREPROCESS_DATA > $DATA_ROOT/pre_cooking.train
    tail -n 3000 $PREPROCESS_DATA > $DATA_ROOT/pre_cooking.valid 
}

function sample_1(){
    # 基础款
    OUTPUT=output/model_cooking_1

    fasttext supervised -input $DATA_ROOT/cooking.train -output $OUTPUT
    
    for(( i=0;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/cooking.valid $i 
    done 
}

function sample_2(){
    #对数据预处理
    OUTPUT=output/model_cooking_2
    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}

function sample_3(){
    # -epoch 25
    OUTPUT=output/model_cooking_3
    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT  -epoch 25
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}

function sample_4(){
    # -lr 1.0  学习率
    OUTPUT=output/model_cooking_4
    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT  -lr 1.0 
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}

function sample_5(){
    # -epoch 25 -lr 1.0
    OUTPUT=output/model_cooking_5
    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT -lr 1.0 -epoch 25
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}

function sample_6(){
    # -wordNgrams 2 
    OUTPUT=output/model_cooking_6
    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT -lr 1.0 -epoch 25 -wordNgrams 2
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}

function sample_7(){
    # -loss hs 分层 softmax
    OUTPUT=output/model_cooking_7

    fasttext supervised -input $DATA_ROOT/pre_cooking.train -output $OUTPUT -lr 1.0 -epoch 25 -wordNgrams 2 -bucket 200000 -dim 50 -loss hs
    
    for(( i=1;i<6;i++ ))
    do
        fasttext test $OUTPUT.bin $DATA_ROOT/pre_cooking.valid $i 
    done
}



# load_data
# sample_1

# # preprocess
# sample_2
# sample_3
# sample_4
# sample_5
sample_6
# sample_7



# fasttext predict output/model_cooking_6.bin -
