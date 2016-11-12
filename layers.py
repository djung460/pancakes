import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_row = np.reshape(x,(N,D))
    out = np.dot(x_row,w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_row = np.reshape(x,(N,D))
    
    dx_row = np.dot(dout,w.T)  # N x D
    dw = np.dot(x_row.T,dout) # D x M
    db = np.sum(dout.T, axis=1) # M x 1
    
    dx = np.reshape(dx_row, x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(0,x)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx = dout
    dx[x <= 0] = 0
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # Get the mean
    batch_mu = 1 / float(N) * np.sum(x,axis=0)
    
    # Get the variance
    batch_var = 1/ float(N) * np.sum((x - batch_mu)**2, axis = 0)
    
    # Normalize the current batch
    x_norm1 = x - batch_mu
    
    sqrt_var = np.sqrt(batch_var + eps)
    x_norm2 = x_norm1 / sqrt_var 
    
    # Scale and shift
    out = gamma * x_norm2 + beta
    
    running_mean = momentum * running_mean + (1.0 - momentum) * batch_mu
    running_var = momentum * running_var + (1.0 - momentum) * batch_var
    
    cache = (x, batch_mu, batch_var, eps, x_norm1, x_norm2, sqrt_var, gamma, beta)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_norm = (x - running_mean) / np.sqrt(running_var + eps)
    
    out = gamma * x_norm + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  x, batch_mu, batch_var, eps, x_norm1, x_norm2, sqrt_var, gamma, beta = cache
  
  N,D = dout.shape
    
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  dx_hat = dout * gamma
    
  dvar = np.sum(dx_hat*x_norm1*-1/2*sqrt_var**(-3), axis = 0)

  dmu = np.sum(dx_hat*-1/sqrt_var,axis=0) + dvar/N*np.sum(-2*x_norm1)
    
  dx = dx_hat/sqrt_var + dvar/N*2*x_norm1 + dmu/N

  dgamma = np.sum(dout * x_norm2,axis=0)
  dbeta = np.sum(dout,axis=0)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  x, batch_mu, batch_var, eps, x_norm1, x_norm2, sqrt_var, gamma, beta = cache
  
  N,D = dout.shape


  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  dx_hat = dout * gamma
    
  dvar = np.sum(dx_hat*x_norm1*-1/2*sqrt_var**(-3), axis = 0)

  dmu = np.sum(dx_hat*-1/sqrt_var,axis=0) + dvar/N*np.sum(-2*x_norm1)
    
  dx = dx_hat/sqrt_var + dvar/N*2*x_norm1 + dmu/N

  dgamma = np.sum(dout * x_norm2,axis=0)
  dbeta = np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.randn(x.shape[0],x.shape[1]) < p).astype(float) / p
    out  = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  p = dropout_param['p']
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    
    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, in_depth, in_width, in_height = x.shape
    num_filters, filter_depth, filter_height, filter_width = w.shape
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # get the conv layer height and width
    conv_height = (in_height - filter_height + 2 * pad) / stride + 1
    conv_width = (in_width - filter_width + 2 * pad) / stride + 1
    
    out = np.zeros((N, num_filters, conv_height, conv_width))
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    npad = ((0,0),(0,0),(pad,pad),(pad,pad))
    x_pad = np.pad(x,npad,mode='constant',constant_values=0)
    # iterate over the examples
    for n in range(N):
        # iterate over the number of filters
        for f in range(num_filters):
            for i in range(conv_height):
                for j in range(conv_width):
                    out[n,f,i,j] = np.sum(
                        x_pad[n, :, i * stride:i * stride + filter_height, j * stride:j * stride + filter_width] * w[f, :]) + b[f]
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x,w,b, conv_param = cache
    
    dx, dw, db = None, None, None
    
    N, in_depth, in_width, in_height = x.shape
    num_filters, filter_depth, filter_height, filter_width = w.shape
    N_dout, dout_depth, dout_height, dout_width = dout.shape
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # get the conv layer height and width
    conv_height = (in_height - filter_height + 2 * pad) / stride + 1
    conv_width = (in_width - filter_width + 2 * pad) / stride + 1
    
    out = np.zeros((N, num_filters, conv_height, conv_width))
    
    npad = ((0,0),(0,0),(pad,pad),(pad,pad))
    
    x_pad = np.pad(x,npad,mode='constant',constant_values=0)
    
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # For dw
    dw = np.zeros((num_filters, filter_depth, filter_height, filter_width))
    for f in range(num_filters):
        # iterate over the number of filters
        for d in range(in_depth):
            for i in range(filter_height):
                for j in range(filter_width):
                    x_pad_window = x_pad[:, d, i:i+dout_height*stride:stride, j:j+dout_width*stride:stride]
                    dw[f,d,i,j] = np.sum(dout[:,f,:, :] * x_pad_window)
    
    # For db
    db = np.zeros((num_filters))
    for f in range(num_filters):
        db[f] = np.sum(dout[:, f, :, :])
        
    # For dx
    dx = np.zeros((N, in_depth, in_height, in_width))
    for n in range(N):
        for i in range(in_height):
            for j in range(in_width):
                for f in range(dout_depth):
                    for k in range(dout_height):
                        for l in range(dout_width):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + pad - k * stride) < filter_height and (i + pad - k * stride) >= 0:
                                mask1[:, i + pad - k * stride, :] = 1.0
                            if (j + pad - l * stride) < filter_width and (j + pad - l * stride) >= 0:
                                mask2[:, :, j + pad - l * stride] = 1.0
                            w_masked = np.sum(
                                w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            
                            dx[n, :, i, j] += dout[n, f, k, l] * w_masked
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    print dx.shape
    
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N, in_depth, in_width, in_height = x.shape
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    HP = (in_height - pool_height) / stride + 1
    WP = (in_width - pool_width) / stride + 1
    
    out = np.zeros((N,in_depth, HP, WP))
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    for n in range(N):
        for d in range(in_depth):
            for i in range(HP):
                for j in range(WP):
                    out[n,d,i,j] = x[n,d,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width].max()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    
    Returns:
    - dx: Gradient with respect to x
    """
    
    x, pool_param = cache
    
    N, in_depth, in_width, in_height = x.shape
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    HP = (in_height - pool_height) / stride + 1
    WP = (in_width - pool_width) / stride + 1
    
    dx = np.zeros(x.shape)
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    for n in range(N):
        for d in range(in_depth):
            for i in range(HP):
                for j in range(WP):
                    x_pool = x[n,d,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
                    max_pool = np.max(x_pool)
                    x_mask = max_pool == x_pool
                    dx[n,d,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width] += dout[n,d,i,j]*x_mask
    return dx

  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  y = y.astype(int)
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
