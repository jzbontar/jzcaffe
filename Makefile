libjzcaffe.so: jzcaffe.cpp
	g++ -I /home/jure/build/caffe/include/ -I /opt/cuda/include/ -lcaffe -lprotobuf -lglog -O2 -shared -fPIC -o libjzcaffe.so jzcaffe.cpp
