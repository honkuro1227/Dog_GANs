#include <math.h>
#include <stdlib.h>

#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a) {
  int i, j;
  for (i = 0; i < m.rows; ++i) {
    double sum = 0;
    for (j = 0; j < m.cols; ++j) {
      double x = m.data[i][j];
      if (a == LOGISTIC) {
        m.data[i][j] = 1.0 / (1.0 + exp(-x));
      } else if (a == RELU) {
        if (x <= 0) m.data[i][j] = 0;
      } else if (a == LRELU) {
        if (x <= 0) m.data[i][j] = 0.1 * x;
      } else if (a == SOFTMAX) {
        m.data[i][j] = exp(m.data[i][j]);
      }
      sum += m.data[i][j];
    }
    if (a == SOFTMAX) {
      for (j = 0; j < m.cols; ++j) {
        m.data[i][j] /= sum;
      }
    }
  }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d) {
  int i, j;
  for (i = 0; i < m.rows; ++i) {
    for (j = 0; j < m.cols; ++j) {
      double x = m.data[i][j];
      if (a == LOGISTIC) {
        d.data[i][j] *= x * (1 - x);
      } else if (a == RELU) {
        if (x <= 0) d.data[i][j] = 0;
      } else if (a == LRELU) {
        if (x <= 0) d.data[i][j] = 0.1;
      }
    }
  }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in) {
  l->in = in;  // Save the input for backpropagation

  // TODO: fix this! multiply input by weights and apply activation function.
  matrix out = matrix_mult_matrix(in, l->w);
  activate_matrix(out, l->activation);

  free_matrix(l->out);  // free the old output
  l->out = out;         // Save the current output for gradient calculation
  return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta) {
  // 1.4.1
  // delta is dL/dy
  // TODO: modify it in place to be dL/d(xw)
  gradient_matrix(l->out, l->activation, delta);

  // 1.4.2
  // TODO: then calculate dL/dw and save it in l->dw
  free_matrix(l->dw);
  matrix in_t = transpose_matrix(l->in);
  matrix dw = matrix_mult_matrix(in_t, delta);  // replace this
  free_matrix(in_t);
  l->dw = dw;

  // 1.4.3
  // TODO: finally, calculate dL/dx and return it.
  matrix w_t = transpose_matrix(l->w);
  matrix dx = matrix_mult_matrix(delta, w_t);  // replace this
  free_matrix(w_t);

  return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay) {
  // TODO:
  // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
  // save it to l->v
  matrix decay_w_and_dw = axpy_matrix(-decay, l->w, l->dw);
  matrix delta_w = axpy_matrix(momentum, l->v, decay_w_and_dw);
  free_matrix(decay_w_and_dw);
  free_matrix(l->v);
  l->v = delta_w;

  // Update l->w
  matrix new_w = axpy_matrix(rate, delta_w, l->w);
  free_matrix(l->w);
  l->w = new_w;

  // Remember to free any intermediate results to avoid memory leaks
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation) {
  layer l;
  l.in = make_matrix(1, 1);
  l.out = make_matrix(1, 1);
  l.w = random_matrix(input, output, sqrt(2. / input));
  l.v = make_matrix(input, output);
  l.dw = make_matrix(input, output);
  l.activation = activation;
  return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X) {
  int i;
  for (i = 0; i < m.n; ++i) {
    X = forward_layer(m.layers + i, X);
  }
  return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL) {
  matrix d = copy_matrix(dL);
  int i;
  for (i = m.n - 1; i >= 0; --i) {
    matrix prev = backward_layer(m.layers + i, d);
    free_matrix(d);
    d = prev;
  }
  free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay) {
  int i;
  for (i = 0; i < m.n; ++i) {
    update_layer(m.layers + i, rate, momentum, decay);
  }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n) {
  if (n <= 0) return -1;
  int i;
  int max_i = 0;
  double max = a[0];
  for (i = 1; i < n; ++i) {
    if (a[i] > max) {
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d) {
  matrix p = forward_model(m, d.X);
  int i;
  int correct = 0;
  for (i = 0; i < d.y.rows; ++i) {
    if (max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols))
      ++correct;
  }
  return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p) {
  int i, j;
  double sum = 0;
  for (i = 0; i < y.rows; ++i) {
    for (j = 0; j < y.cols; ++j) {
      sum += -y.data[i][j] * log(p.data[i][j]);
    }
  }
  return sum / y.rows;
}

// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate,
                 double momentum, double decay) {
  int e;
  for (e = 0; e < iters; ++e) {
    data b = random_batch(d, batch);
    matrix p = forward_model(m, b.X);
    fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
    matrix dL = axpy_matrix(-1, p, b.y);  // partial derivative of loss dL/dy
    backward_model(m, dL);
    update_model(m, rate / batch, momentum, decay);
    free_matrix(dL);
    free_data(b);
  }
}

// Questions
//
// 5.2.2.1 Why might we be interested in both training accuracy and
// testing accuracy? What do these two numbers tell us about our current
// model?
// They both says how good our model is. The training accuracy tells us
// how good our model is on the training dataset. The testing accuracy
// tells us how good our model is with data outside our training set.
// We can use it to determine if our model is over-fit to the training data
// or not.
//
// 5.2.2.2 Try varying the model parameter for learning rate to different
// powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the
// model. What patterns do you see and how does the choice of learning
// rate affect both the loss during training and the final model accuracy?
// rate=10:
//   - during training: losses are nan (it overshoots)
//   - final result: both accuracies are ~0.1
// rate=1:
//   - during training: losses looks normal but looks too high
//   - final result: both accuracies are ~0.85 (still overshoots)
// rate=0.1:
//   - during training: losses looks normal
//   - final result: both are ~0.91
// rate=0.01:
//   - during training: losses went down slowly
//   - final result: both are ~0.90 (the program stop before the loss
//                   reaches the global minimum)
// rate=0.001:
//   - during training: losses went down more slowly
//   - final result: both are ~0.85 (the program stop before the loss
//                   reaches the global minimum)
//
// the higher ther learning rate, the faster the training loss to hit
// the global minima. However, if it is too high, it could overshoot
// and misses the global minima.
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers
// of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay
// affect the final model training and test accuracy?
// if the decay is 0, this means it does not decay.
//   - it has a good result, but could cause the model to overfit
// if the decay is 1, this means decays the same amount as the weight
//   - it could cause the model to not reach the global minima
//
// 5.2.3.1 Currently the model uses a logistic activation for the first
// layer. Try using a the different activation functions we programmed.
// How well do they perform? What's best?
// for training loss
// Relu: 0.95
// Leaky Relu: 0.89
// Logistic: 0.94
// Relu works the best
//
// 5.2.3.2 Using the same activation, find the best (power of 10) learning
// rate for your model. What is the training accuracy and testing
// accuracy?
// I found that learning rate works the best at 0.1
// training accuracy is 0.951 and testing accuracy is 0.943
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try
// adding some decay to your model. What happens, does it help? Why or why
// not may this be?
// I found that decay at 0.01 works the best. The training accuracy decrease
// to 0.949, but the testing accuracy increase to 0.945
// This helps increasing the testing accuracy because the weight decay tries
// to not overfit the model to the training data by decreasing the weight
// every update. So, the model is more generalizable.
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers
// should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify
// your model to train for 3000 iterations instead of 1000. Look at the
// training and testing error for different values of decay (powers of 10,
// 10^-4 -> 10^0). Which is best? Why?
// decay=1:
//   - training accuracy: 0.937
//   - testing accuracy: 0.936
// decay=0.1:
//   - training accuracy: 0.969
//   - testing accuracy: 0.962
// decay=0.01:
//   - training accuracy: 0.968
//   - testing accuracy: 0.956
// decay=0.001:
//   - training accuracy: 0.976
//   - testing accuracy: 0.963
// decay=0.0001:
//   - training accuracy: 0.979
//   - testing accuracy: 0.966
// decay at 0.0001 works the best since the more we decay the model, the harder
// it is for the weight to make the loss reaches the global minima. So, we need
// a decay to regularize but not too much.
//
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// my learning rate is 0.005, and here is my network layout:
//   make_layer(inputs, 64, RELU),
//   make_layer(64, 64, RELU),
//   make_layer(64, outputs, SOFTMAX)
// as a final result the training accuracy is 0.471, and the testing accuracy is
// 0.459
//