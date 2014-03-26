libjzcaffe.so: jzcaffe.cpp
	g++ -I ../include/ -I /opt/cuda/include/ -lcaffe -lprotobuf -lglog -shared -fPIC -o libjzcaffe.so jzcaffe.cpp
