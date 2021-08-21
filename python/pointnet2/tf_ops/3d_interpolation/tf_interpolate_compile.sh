# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB
echo "SPAM"

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda/include \
#-I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ \
#-L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include \
-I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -l:libtensorflow_framework.so.1 -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
