import numpy as np

def _dim(x,dim):
    if dim == 1:
        return (x,1)
    return(x,x)

class my_Conv2D:
    def __init__(self,in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0, bias=False, dim = 2, dilation=1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _dim(kernel_size, dim)
        self.stride = _dim(stride, dim)
        self.padding = _dim(padding, dim)
        self.dilation = _dim(1)
        self.groups = 1
        self.bias = bias

def conv2d(input_data, kernel, bias):
    input_shape = input_data.shape
    kernel_shape = kernel.shape
    output_shape = (input_shape[0] - kernel_shape[0] + 1, input_shape[1] - kernel_shape[1] + 1, kernel_shape[3])
    output_data = np.zeros(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                output_data[i, j, k] = np.sum(input_data[i:i+kernel_shape[0], j:j+kernel_shape[1], :] * kernel[:, :, :, k]) + bias[k]
    return output_data

input_data = np.random.rand(10, 10, 1)
kernel = np.random.rand(3, 3, 1, 4)
bias = np.random.rand(4)
output_data = conv2d(input_data, kernel, bias)
output_data = conv2d(output_data, kernel, bias)
output_data = conv2d(output_data, kernel, bias)
output_data = conv2d(output_data, kernel, bias)
print(output_data)
