import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

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
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N = x.shape[0]
  F = w.shape[0]
  H = x.shape[2]
  W = x.shape[3]
  HH = w.shape[2]
  WW = w.shape[3]
  C = x.shape[1]
  H_prime = 1 + (H + 2 * pad - HH) // stride
  W_prime = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, H_prime, W_prime))
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
  w_reshape = w.reshape(F, C*HH*WW)
  x_reshape = np.zeros((C*HH*WW, H_prime*W_prime))
  for i in range(N):
    node = 0
    for j in np.arange(0, xpad.shape[2]-HH+1, stride):
      for k in np.arange(0, xpad.shape[3]-WW+1, stride):
          x_reshape[:, node] = xpad[i,:,j:j+HH,k:k+WW].reshape(C*HH*WW)
          node = node + 1
    out[i] = (np.dot(w_reshape, x_reshape) + b.reshape((F,1))).reshape(F, H_prime, W_prime)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
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
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, C, padded_H, padded_W = xpad.shape
  dx = np.zeros((N, C, H, W))
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  w_reshape = w.reshape(F, C*HH*WW)
  x_reshape = np.zeros((C*HH*WW, out_height * out_width))
  for i in range(N):
    out_reshape = dout[i].reshape(F, out_height * out_width)
    w_temp = np.dot(w_reshape.T, out_reshape)
    dx_temp = np.zeros((C, padded_H, padded_W))
    node = 0
    for j in range(0, padded_H-HH+1, stride):
        for k in range(0, padded_W-WW+1, stride):
            dx_temp[:,j:j+HH,k:k+WW] = dx_temp[:,j:j+HH,k:k+WW] + w_temp[:,node].reshape(C,HH,WW)
            x_reshape[:,node] = xpad[i,:,j:j+HH,k:k+WW].reshape(C*HH*WW)
            node = node + 1
    dx[i] = dx_temp[:,pad:-pad,pad:-pad]
    dw = dw + np.dot(out_reshape, x_reshape.T).reshape(F,C,HH,WW)
    db = db + np.sum(out_reshape, axis=1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

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
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_prime = 1 + (H - pool_height) // stride
  W_prime = 1 + (W - pool_width) // stride
  out = np.zeros((N, C, H_prime, W_prime))
  x_reshape = np.zeros((C*H*W, H_prime*W_prime))
  for i in range(N):
    out_region = np.zeros((C, H_prime * W_prime))
    node = 0
    for j in np.arange(0, H-pool_height+1, stride):
      for k in np.arange(0, W-pool_width+1, stride):
          x_down = x[i,:,j:j+pool_height,k:k+pool_width].reshape(C, pool_height * pool_width)
          out_region[:, node] = np.max(x_down, axis=1)
          node = node + 1
    out[i] = out_region.reshape(C, H_prime, W_prime)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
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
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  N, C, H_prime, W_prime = dout.shape
  dx = np.zeros((N, C, H, W))
  for i in range(N):
    out_reshape = dout[i].reshape(C, H_prime * W_prime)
    node = 0
    for j in range(0, H-pool_height+1, stride):
        for k in range(0, W-pool_width+1, stride):
            x_down = x[i,:,j:j+pool_height,k:k+pool_width].reshape(C, pool_height * pool_width)
            indices = np.argmax(x_down, axis=1)
            out_region = out_reshape[:,node]
            node = node + 1
            d_pool = np.zeros(x_down.shape)
            channels = np.arange(C)
            d_pool[channels, indices] = out_region
            dx[i,:,j:j+pool_height,k:k+pool_width] = dx[i,:,j:j+pool_height,k:k+pool_width] + d_pool.reshape(C, pool_height, pool_width)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x_reshape = x.reshape((N*H*W, C))
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x_reshape.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x_reshape.dtype))
    
  sample_mean = x_reshape.mean(axis=0)
  sample_var = x_reshape.var(axis=0) + eps
  std = np.sqrt(sample_var)
  x_hat = (x_reshape - sample_mean) / std
  out = gamma * x_hat + beta
  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var
  cache = {'x':x_reshape, 'mean':sample_mean, 'var':sample_var, 'std':std, 'x_hat':x_hat, 'gamma':gamma, 'beta':beta}  
  out = out.reshape((N, C, H, W))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm backward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = dout.reshape((N*H*W, C))

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * cache['x_hat'], axis=0)
    
  D = dout.shape[0]
  dx_hat = dout * cache['gamma']
  dsample_var = np.sum((cache['x'] - cache['mean']) * dx_hat, axis=0) * (-1 / (2 * (cache['var'] ** (3/2))))
  dsample_mean = -((np.sum(dx_hat, axis=0) / cache['std']) + (dsample_var * (2/D) * np.sum(cache['x'] - cache['mean'], axis = 0)))
  dx = (dx_hat / cache['std']) + (((2 * (cache['x'] - cache['mean'])) / D) * dsample_var) + (dsample_mean / D)
  dx = dx.reshape((N, C, H, W))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta