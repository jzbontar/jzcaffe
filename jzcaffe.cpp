#include <stdio.h>
#include "caffe/caffe.hpp"

/*** Blob ***/
extern "C" caffe::Blob<float> *blob_new(int num, int channels, int height, int width)
{
	caffe::Blob<float> *blob = new caffe::Blob<float>(num, channels, height, width);
	return blob;
}

extern "C" void blob_free(caffe::Blob<float> *blob)
{
	delete blob;
}

extern "C" void blob_print(caffe::Blob<float> *blob)
{
	int i;
	const float *data = blob->cpu_data();

	for (i = 0; i < blob->count(); i++) {
		printf("%d %f\n", i, data[i]);
	}
}

extern "C" int blob_num(caffe::Blob<float> *blob) { return blob->num(); }
extern "C" int blob_channels(caffe::Blob<float> *blob) { return blob->channels(); }
extern "C" int blob_height(caffe::Blob<float> *blob) { return blob->height(); }
extern "C" int blob_width(caffe::Blob<float> *blob) { return blob->width(); }
extern "C" int blob_count(caffe::Blob<float> *blob) { return blob->count(); }
extern "C" float *blob_cpu_data(caffe::Blob<float> *blob) { return blob->mutable_cpu_data(); }

/*** Layer ***/
struct Layer {
	vector<struct caffe::Blob<float>*> bottom;
	vector<struct caffe::Blob<float>*> top;
	caffe::Layer<float>* layer;
};

Layer *layer_new(caffe::Layer<float>* caffe_layer, caffe::Blob<float> *bottom)
{
	Layer *layer = new Layer();
	caffe::Blob<float>* top = new caffe::Blob<float>();

	layer->bottom.push_back(bottom);
	layer->top.push_back(top);

	layer->layer = caffe_layer;
	caffe_layer->SetUp(layer->bottom, &(layer->top));

	return layer;
}

extern "C" void layer_forward(Layer *layer)
{
	layer->layer->Forward(layer->bottom, &(layer->top));
}

extern "C" void layer_backward(Layer *layer, bool propagate_down)
{
	layer->layer->Backward(layer->bottom, propagate_down, &(layer->top));
}

extern "C" void layer_free(Layer *layer)
{
	for (int i = 0; i < layer->top.size(); i++) {
		delete layer->top[i];
	}
	delete layer->layer;
	delete layer;
}

/*** Inner Product Layer ***/
extern "C" Layer *inner_product_layer_new(caffe::Blob<float> *bottom, int num_output)
{
	caffe::LayerParameter layer_param;
	layer_param.set_num_output(num_output);
	return layer_new(new caffe::InnerProductLayer<float>(layer_param), bottom);
}

/*** Tanh ***/
extern "C" Layer *tanh_layer_new(caffe::Blob<float> *bottom)
{
	caffe::LayerParameter layer_param;
	return layer_new(new caffe::TanHLayer<float>(layer_param), bottom);
}
