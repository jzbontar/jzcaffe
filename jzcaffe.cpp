#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"

__attribute__((constructor)) void init() {
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

/*** util ***/
extern "C" void deviceSynchronize()
{
	cudaDeviceSynchronize();
}

/*** Blob ***/
extern "C" caffe::Blob<float> *blob(int num, int channels, int height, int width)
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
	const float *diff = blob->cpu_diff();

	printf("%dx%dx%dx%d\n", blob->num(), blob->channels(), blob->height(), blob->width());
	for (i = 0; i < blob->count() && i < 10; i++) {
		printf("% f % f\n", data[i], diff[i]);
	}

	if (i == 10) {
		printf("...\n");
	}
}

extern "C" void blob_host2device(caffe::Blob<float> *blob, float *host_data, int diff)
{
	float *device_data = diff ? blob->mutable_gpu_diff() : blob->mutable_gpu_data();
	cudaMemcpy(device_data, host_data, sizeof(float) * blob->count(), cudaMemcpyHostToDevice);
}

extern "C" void blob_device2host(caffe::Blob<float> *blob, float *host_data, int diff)
{
	float *device_data = diff ? blob->mutable_gpu_diff() : blob->mutable_gpu_data();
	cudaMemcpy(host_data, device_data, sizeof(float) * blob->count(), cudaMemcpyDeviceToHost);
}

extern "C" int blob_num(caffe::Blob<float> *blob) { return blob->num(); }
extern "C" int blob_channels(caffe::Blob<float> *blob) { return blob->channels(); }
extern "C" int blob_height(caffe::Blob<float> *blob) { return blob->height(); }
extern "C" int blob_width(caffe::Blob<float> *blob) { return blob->width(); }
extern "C" int blob_count(caffe::Blob<float> *blob) { return blob->count(); }

/*** Layer ***/
struct Layer {
	vector<caffe::Blob<float>*> bottom;
	vector<caffe::Blob<float>*> top;
	caffe::Layer<float>* layer;
};

Layer *layer(caffe::Layer<float>* caffe_layer, caffe::Blob<float> *bottom)
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
extern "C" caffe::Blob<float> *layer_blobs(Layer *layer, int i) { 
	return layer->layer->blobs()[i].get(); }

extern "C" float layer_forward(Layer *layer)
{
	return layer->layer->Forward(layer->bottom, &layer->top);
}

extern "C" void layer_backward(Layer *layer, bool propagate_down)
{
	layer->layer->Backward(layer->top, propagate_down, &layer->bottom);
}

extern "C" void layer_free(Layer *layer)
{
	for (unsigned int i = 0; i < layer->top.size(); i++) {
		delete layer->top[i];
	}
	delete layer->layer;
	delete layer;
}

extern "C" void layer_update_parameters(Layer *layer, float learning_rate)
{
	assert(caffe::Caffe::mode() == caffe::Caffe::GPU);

	vector<boost::shared_ptr<caffe::Blob<float> > >& layer_blobs = layer->layer->blobs();
	for (unsigned int i = 0; i < layer_blobs.size(); i++) {
		caffe::caffe_gpu_axpy(layer_blobs[i]->count(), -learning_rate,
			layer_blobs[i]->gpu_diff(), layer_blobs[i]->mutable_gpu_data());
	}
}

/*** Inner Product Layer ***/
extern "C" Layer *inner_product_layer(caffe::Blob<float> *bottom, int num_output)
{
	caffe::LayerParameter layer_param;
	layer_param.set_num_output(num_output);
	layer_param.mutable_weight_filler()->set_type("xavier");
	layer_param.mutable_bias_filler()->set_type("constant");
	return layer(new caffe::InnerProductLayer<float>(layer_param), bottom);
}

/*** Conv Layer ***/
extern "C" Layer *conv_layer(caffe::Blob<float> *bottom, int num_output, int kernel_size, int stride)
{
	caffe::LayerParameter layer_param;
	layer_param.set_num_output(num_output);
	layer_param.set_kernelsize(kernel_size);
	layer_param.set_stride(stride);
	layer_param.mutable_weight_filler()->set_type("xavier");
	layer_param.mutable_bias_filler()->set_type("constant");
	return layer(new caffe::InnerProductLayer<float>(layer_param), bottom);
}

/*** Pooling ***/
extern "C" Layer *pooling_layer(caffe::Blob<float> *bottom, char *type, int kernel_size, int stride)
{
	caffe::LayerParameter_PoolMethod itype;
	caffe::LayerParameter layer_param;

	if (strcmp(type, "max")) {
		itype = caffe::LayerParameter_PoolMethod_MAX;
	} else if (strcmp(type, "ave")) {
		itype = caffe::LayerParameter_PoolMethod_AVE;
	} else if (strcmp(type, "stochastic")) {
		itype = caffe::LayerParameter_PoolMethod_STOCHASTIC;
	} else {
		assert(0);
	}
	layer_param.set_pool(itype);
	layer_param.set_kernelsize(kernel_size);
	layer_param.set_stride(stride);

	return layer(new caffe::PoolingLayer<float>(layer_param), bottom);
}


/*** Tanh ***/
extern "C" Layer *tanh_layer(caffe::Blob<float> *bottom)
{
	caffe::LayerParameter layer_param;
	return layer(new caffe::TanHLayer<float>(layer_param), bottom);
}

/*** Softmax Loss ***/
extern "C" Layer *softmax_with_loss_layer(caffe::Blob<float> *prediction, caffe::Blob<float> *target)
{
	Layer *layer = new Layer();

	layer->bottom.push_back(prediction);
	layer->bottom.push_back(target);

	caffe::LayerParameter layer_param;
	layer->layer = new caffe::SoftmaxWithLossLayer<float>(layer_param);
	layer->layer->SetUp(layer->bottom, &(layer->top));

	return layer;
}
