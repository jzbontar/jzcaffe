from ctypes import CDLL, c_int, c_int64, c_float, c_void_p, py_object, pythonapi
import numpy as np

jz = CDLL('./libjzcaffe.so')
jz.layer_update_parameters.argtypes = [c_void_p, c_float]
jz.blob_host2device.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]
jz.blob_device2host.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]

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
    action = 'mlp'

    if action == 'foo':
        import pyprof

        input = jz.blob(128,3,100,100)
        labels = jz.blob(128,1,1,1)

        net = jz.conv_layer(input, 96, 11, 1)
        jz.layer_forward(net)
        jz.layer_backward(net, True)

        for i in range(3):
            pyprof.tic('forward/backward')
            pyprof.tic('forward')
            jz.layer_forward(net)
            jz.deviceSynchronize()
            pyprof.toc('forward')

            pyprof.tic('backward')
            jz.layer_backward(net, True)
            jz.deviceSynchronize()
            pyprof.toc('backward')
            pyprof.toc('forward/backward')
        pyprof.dump()

        #net = Sequential(input)
        #net.add(jz.inner_product_layer,  10)
        #net.add(jz.tanh_layer)
        #net.add(jz.softmax_with_loss_layer, labels)

        #net.forward()

    if action == 'mlp':
        import sys
        import pydatasets

        learning_rate = 0.1
        batch_size = 128

        X, y = pydatasets.mnist()
        X.resize(70000, 1, 28, 28)

        x_batch = jz.blob(batch_size, 1, 28, 28)
        y_batch = jz.blob(batch_size, 1, 1, 1)
        net_output = np.array((batch_size, 10), dtype=np.float32)

        net = Sequential(x_batch)
        net.add(jz.inner_product_layer, 800)
        net.add(jz.tanh_layer)
        net.add(jz.inner_product_layer, 10)
        net.add(jz.softmax_with_loss_layer, y_batch)


        for epoch in range(10):
            # train
            #for i in range(0, 60000 - batch_size, batch_size):
            #    jz.blob_host2device(x_batch, X[i:i + batch_size])
            #    jz.blob_host2device(y_batch, y[i:i + batch_size])
            #    
            #    net.forward()
            #    net.backward()
            #    net.update_parameters(learning_rate)

            # test
            for i in range(60000, 70000 - batch_size, batch_size):
                jz.blob_host2device(x_batch, X[i:i + batch_size])
                
                net.forward()

                out_blob = jz.layer_top(net.layers[-2], 0)
                #jz.blob_device2host(out_blob, net_output)

                sys.exit()


            print epoch
