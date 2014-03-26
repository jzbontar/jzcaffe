from ctypes import CDLL, c_int, c_float, c_void_p, py_object, pythonapi
import numpy as np

jz = CDLL('./libjzcaffe.so')
jz.layer_update_parameters.argtypes = [c_void_p, c_float]

def blob2np(blob):
    mem = jz.blob_cpu_data(blob)
    buffer_from_memory = pythonapi.PyBuffer_FromReadWriteMemory
    buffer_from_memory.restype = py_object
    buffer = buffer_from_memory(mem, 4 * jz.blob_count(blob))
    data = np.frombuffer(buffer, np.float32)
    data.reshape(
        jz.blob_num(blob), 
        jz.blob_channels(blob), 
        jz.blob_width(blob), 
        jz.blob_height(blob))
    return data

class Sequential:
    def __init__(self, input):
        self.input = input
        self.layers = []

    def add(self, layer, *args):
        input = jz.layer_top(self.layers[-1], 0) if self.layers else self.input
        self.layers.append(layer(input, *args))

    def forward(self):
        for layer in self.layers:
            jz.layer_forward(layer)
    
    def backward(self):
        for layer in self.layers:
            jz.layer_backward(layer, 1)
    
    def update_parameters(self, learning_rate):
        for layer in self.layers:
            jz.layer_update_parameters(layer, learning_rate)
    

if __name__ == '__main__':
    input = jz.blob(128,3,100,100)
    labels = jz.blob(128,1,1,1)

    net = Sequential(input)
    net.add(jz.inner_product_layer,  10)
    net.add(jz.tanh_layer)
    net.add(jz.softmax_with_loss_layer, labels)

    net.forward()
