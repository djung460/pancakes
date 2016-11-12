import numpy as np

from layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    in_depth, in_height, in_width = input_dim
    stride_conv = 1
    pad = (filter_size - 1) / 2
    
    conv_height = (in_height - filter_size + 2 * pad) / stride_conv + 1
    conv_width = (in_width - filter_size + 2 * pad) / stride_conv + 1
    
    # Weights and biases for convolution layer
    self.params['W1'] = weight_scale*np.random.randn(num_filters,in_depth,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    
    # Pooling parameters
    pool_depth = num_filters
    pool_height = 2
    pool_width = 2
    pool_stride = 2
    # Find shape after pooling
    pool_width = (conv_width - pool_width) / pool_stride + 1
    pool_height = (conv_height - pool_height) / pool_stride + 1
    
    # Next two for the affine layers
    self.params['W2'] = weight_scale*np.random.randn(pool_depth*pool_width*pool_height,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    reg = self.reg
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    h1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    h2, cache2 = affine_relu_forward(h1, W2, b2)
    scores, cache3 = affine_forward(h2,W3,b3)
    
    if y is None:
      return scores

    loss, dout = softmax_loss(scores, y)
    
    loss += 0.5*reg * np.sum(W1**2) + 0.5*reg * np.sum(W2**2) + 0.5*reg * np.sum(W3**2)
    
    grads = {}

    dx3,dw3,db3 = affine_backward(dout,cache3)
    dw3 += reg * W3
    dx2,dw2,db2 = affine_relu_backward(dx3, cache2)
    dw2 += reg * W2
    dx1,dw1,db1 = conv_relu_pool_backward(dx2,cache1)
    dw1 += reg * W1
    
    grads.update({'W1': dw1,
                      'b1': db1,
                      'W2': dw2,
                      'b2': db2,
                      'W3': dw3,
                      'b3': db3})
    return loss, grads
  
  
pass
