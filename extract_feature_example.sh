#!/usr/bin/env sh 
# args for EXTRACT_FEATURE 
TOOL=./build/tools 
MODEL=./models/autodial/alexnet/trained_model_iter_2000.caffemodel #下载得到的caffe model 
PROTOTXT=./models/autodial/alexnet/auto/deploy.prototxt # 网络定义
LAYER=fc8 # 提取层的名字，如提取fc7等
LEVELDB=./models/autodial/features_fc8s # 保存的leveldb路径

BATCHSIZE=795


# args for LEVELDB to MAT 
DIM=31 # 需要手工计算feature长度 
OUT=./models/autodial/features_fc8s.mat #.mat文件保存路径 
BATCHNUM=1 # 有多少个batch， 本例只有两张图， 所以只有一个batch

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE lmdb GPU
python lmdb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT
