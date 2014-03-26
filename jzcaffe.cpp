#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"

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
	vector<caffe::Blob<float>*> bottom;
	vector<caffe::Blob<float>*> top;
	caffe::Layer<float>* layer;
};

Layer *layer_new(caffe::Layer<float>* caffe_layer, caffe::Blob<float> *bottom)
{
	Layer *layer = new Layer();
	caffe::Blob<float>* top = new caffe::Blob<float>();

	layer->bottom.push_back(bottom);
	layer->top.push_back(top);

	layer->layer = caffe_layer;
	layer->layer->SetUp(layer->bottom, &(layer->top));

	return layer;
}

extern "C" caffe::Blob<float> *layer_bottom(Layer *layer, int i) { return layer->bottom[i]; }
extern "C" caffe::Blob<float> *layer_top(Layer *layer, int i) { return layer->top[i]; }

extern "C" void layer_forward(Layer *layer)
{
	layer->layer->Forward(layer->bottom, &layer->top);
}

extern "C" void layer_backward(Layer *layer, bool propagate_down)
{
	layer->layer->Backward(layer->top, propagate_down, &layer->bottom);
}

extern "C" void layer_free(Layer *layer)
{
	for (int i = 0; i < layer->top.size(); i++) {
		delete layer->top[i];
	}
	delete layer->layer;
	delete layer;
}

extern "C" void layer_update_parameters(Layer *layer, float learning_rate)
{
	switch (caffe::Caffe::mode()) {
	case caffe::Caffe::CPU:
		//TODO
		break;
	case caffe::Caffe::GPU:
		vector<boost::shared_ptr<caffe::Blob<float> > >& layer_blobs = layer->layer->blobs();
		for (int i = 0; i < layer_blobs.size(); i++) {
			caffe::caffe_gpu_axpy(layer_blobs[i]->count(), learning_rate,
				layer_blobs[i]->gpu_diff(), layer_blobs[i]->mutable_gpu_data());
		}
		break;
	}
}

/*** Inner Product Layer ***/
extern "C" Layer *inner_product_layer_new(caffe::Blob<float> *bottom, int num_output)
{
	caffe::LayerParameter layer_param;
	layer_param.set_num_output(num_output);
	return layer_new(new caffe::InnerProductLayer<float>(layer_param), bottom);
}

/*** Conv Layer ***/
extern "C" Layer *conv_layer_new(caffe::Blob<float> *bottom, int num_output, int kernel_size, int stride)
{
	caffe::LayerParameter layer_param;
	layer_param.set_num_output(num_output);
	layer_param.set_kernelsize(kernel_size);
	layer_param.set_stride(stride);
	return layer_new(new caffe::InnerProductLayer<float>(layer_param), bottom);
}

/*** Tanh ***/
extern "C" Layer *tanh_layer_new(caffe::Blob<float> *bottom)
{
	caffe::LayerParameter layer_param;
	return layer_new(new caffe::TanHLayer<float>(layer_param), bottom);
}

/*** Softmax Loss ***/
extern "C" Layer *softmax_with_loss_layer_new(caffe::Blob<float> *prediction, caffe::Blob<float> *target)
{
	Layer *layer = new Layer();

	layer->bottom.push_back(prediction);
	layer->bottom.push_back(target);

	caffe::LayerParameter layer_param;
	layer->layer = new caffe::SoftmaxWithLossLayer<float>(layer_param);
	layer->layer->SetUp(layer->bottom, &(layer->top));

	return layer;
}
