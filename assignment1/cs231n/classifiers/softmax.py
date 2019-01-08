import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    for j in range(C):
      if j == y[i]:
        dW[:,j] += (exp_scores[j] / sum_exp_scores - 1) * X[i] 
      else:
        dW[:,j] += (exp_scores[j] / sum_exp_scores) * X[i]
    loss += - np.log(exp_scores[y[i]] / sum_exp_scores)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  C = W.shape[1]
  N = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # N x C
  exp_scores = np.exp(scores) # N x C
  sum_exp_scores = np.sum(exp_scores, axis = 1) # N x 1
  L_i = - np.log(exp_scores[range(N),y] / sum_exp_scores) # N x 1
  loss = np.sum(L_i) / N 
  loss += 0.5 * reg * np.sum(W * W)
  
  indicator = exp_scores.T / sum_exp_scores # C x N
  indicator[y,range(N)] -= 1
  dW = indicator.dot(X).T
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

