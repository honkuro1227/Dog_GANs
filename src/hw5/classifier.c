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
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                // TODO


                    d.data[i][j]=d.data[i][j] *x*(1-x);


            } else if (a == RELU){
                // TODO
                if(x>0){
                    d.data[i][j]=d.data[i][j];
                }
                else{
                    d.data[i][j]=0;
                }

            } else if (a == LRELU){
                // TODO
                if(x>0){
                    d.data[i][j]=d.data[i][j] ;
                }
                else{
                    d.data[i][j]=0.1*d.data[i][j];
                }
            }

        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix w=l->w;
    ACTIVATION a=l->activation;
    matrix out=matrix_mult_matrix(l->in,w);
    //matrix out = make_matrix(in.rows, l->w.cols);
    activate_matrix(out,a);
    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    //matrix w=l->w;

    matrix w=l->w;
    matrix x=l->in;
    ACTIVATION a=l->activation;
    matrix xw=matrix_mult_matrix(x,w);
    gradient_matrix(xw,a,delta);


    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix xt = transpose_matrix(x);
    matrix dw = matrix_mult_matrix(xt,delta); // replace this
    l->dw = dw;
    free_matrix(xt);

    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix wt=transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta,wt); // replace this
    free_matrix(wt);
    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix w=l->w;
    matrix delta_w_t_1=l->v;
    matrix dw=l->dw;
    matrix dw_momentum=axpy_matrix(momentum,delta_w_t_1,dw);
    matrix current_w=axpy_matrix(-1*decay,w,dw_momentum);
    free_matrix(l->v);
    l->v=current_w;
    // Update l->w
    matrix w_t_plus1=axpy_matrix(rate,current_w,w);
    free_matrix(l->w);
    l->w=w_t_plus1;


    // Remember to free any intermediate results to avoid memory leaks

}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
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
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
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
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double) correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// Because we would like to realize performance of model, the accuracy can show underfit or overfit of model.
//
// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// rate=10:
//    the weight will update divergent, it will make weight jump too fast to catch the minimum
//    as result: accuracy both model~0.1
// rate=10^0:
//   Much better than ten, but still too large for model.
//   final result: both are ~0.88,
// rate=10^-1:
//   The highest accuracy of these test, NN can find the minimum value
//   final result: both are ~0.92, converge much faster
// rate=10^-2:
//   pattern is similar rate =0.1. However, the accuracy is poorer than 0.1, because it is too slow to find minimum.
//   accuracy: train model is ~0.903, test model is~0.901,
// rate=10^-3:
//   because it update too slow, we cannot reach minimum before iteration stop.
//   - final result: train model=0.86 test model =0.86
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// We want to avoid too large updating weight which might cause overfitting, so we design decay to penalty large weight update.
// When decay is 1, the accuracy of train model is 0.89785 and test model is 0.9056.
// When the decay is closer zero, the accuracy of training model becomes higher. However, the testing accuracy does not much greater than large decay.
// Moreover, there are similar result, when decay smaller than 0.1. I guess it might due to testing is similar with training data, so decay is not helpful here.
//
// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// Relu    train model:0.901 test model:0.9092
// LRelu   train model:0.903 test model:0.9098
// softmax train model:0.901 test model:0.9092
// LRelu is best one.
// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// learning rate=0.1, accuracy = training ~0.92 test ~0.92
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// decay=1, accuracy of train model is 0.9046, testing model is 0.9074
// decay=0, accuracy of train model is 0.9167, testing model is 0.9196
// if decay =1, means large penalty on update weight. It might lead iterations cannot reach minimum.
// on the other hand, if decay=0, means no penalty on update weight. It might lead model overfit.
// However, In this case, data is not big different between training and testing data, so it seems decay =0 is good choice.
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// decay=0, train accuracy:0.9258,test accuracy: 0.9224
// decay=1, train accuracy:0.9048,test accuracy: 0.8998
// decay=0.1, train accuracy:0.9196,test accuracy: 0.9204
// decay=0.01, train accuracy:0.9249,test accuracy: 0.9225
// decay=0.001, train accuracy:0.9258,test accuracy: 0.9223
// decay=0.0001, train accuracy:0.9258,test accuracy: 0.9223
// decay=0.01 is the best. We can see train accuracy is lower than decay=0,but test accuracy is better than decay=0.
// because it avoid overfitting of model.
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// training accuracy: 0.3524
// test accuracy:     0.3494
// setting: batch = 128,iters = 3000,rate = .0001,momentum = .9,decay = 0.0005
// layer=
//[ make_layer(inputs, 64, LRELU),
//  make_layer(64, 64, LRELU),
//  make_layer(64, 32, LRELU),
//  make_layer(32, outputs, SOFTMAX)]

