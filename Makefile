libjzcaffe.so: jzcaffe.cpp
	g++ -I ../include/ -I /opt/cuda/include/ -lcaffe -lprotobuf -lglog -O2 -shared -fPIC -o libjzcaffe.so jzcaffe.cpp
