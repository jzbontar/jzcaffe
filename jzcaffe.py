from ctypes import CDLL, c_int, c_void_p, py_object, pythonapi
import numpy as np

jz = CDLL('./libjzcaffe.so')

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

if __name__ == '__main__':
    input = jz.blob_new(128,1,28,28)
    labels = jz.blob_new(128,1,1,1)

    net1 = jz.inner_product_layer_new(input, 10)
    net2 = jz.tanh_layer_new(jz.layer_top(net1, 0))
    net3 = jz.softmax_with_loss_layer_new(jz.layer_top(net2, 0), labels)

    for i in range(50000):
        jz.layer_forward(net1)
        jz.layer_forward(net2)
        jz.layer_forward(net3)

        jz.layer_backward(net3, 1)
        jz.layer_backward(net2, 1)
        jz.layer_backward(net1, 0)

