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

import time
if __name__ == '__main__':
    input = jz.blob(2,3,4,5)

    net1 = jz.inner_product(input, 10)

    for i in range(10000000):
        jz.layer_forward(net1)


    jz.layer_free(net1)
    jz.blob_free(input)
