#/bin/bash

echo "bacon"

nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

echo "eggs"

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB
echo "SPAM"

TF_BASE_PATH=/home/solomon/.local/lib/python3.8/site-packages

# TF1.4
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
#-I $TF_BASE_PATH/tensorflow/include -I /usr/local/cuda/include -I $TF_BASE_PATH/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ \
#-O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB -L/home/solomon/.local/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.1

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include \
-I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -l:libtensorflow_framework.so.1 -O2 