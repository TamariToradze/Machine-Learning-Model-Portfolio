from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layer_sizes = [input_dim] + hidden_dims + [num_classes]
        layer_sizes = [input_dim, *hidden_dims, num_classes]

        if self.normalization == 'batchnorm':
          for idx in range(1, len(layer_sizes) - 1):
            w_key = f"W{idx}"
            b_key = f"b{idx}"
            self.params[w_key] = np.random.normal(0, weight_scale, (layer_sizes[idx - 1], layer_sizes[idx]))
            self.params[b_key] = np.zeros(layer_sizes[idx])
           
            gammak = f"gamma{idx}"
            betak = f"beta{idx}"
            self.params[gammak] = np.ones((layer_sizes[idx],))
            self.params[betak] = np.zeros((layer_sizes[idx],))
          self.params[f"W{len(layer_sizes) - 1}"] = np.random.normal(0, weight_scale, (layer_sizes[-2], layer_sizes[-1]))
          self.params[f"b{len(layer_sizes) - 1}"] = np.zeros(layer_sizes[-1])  
        else:    
          for idx in range(1, len(layer_sizes)):
            w_key = f"W{idx}"
            b_key = f"b{idx}"
            lind1=layer_sizes[idx - 1]
            lind=layer_sizes[idx]
            self.params[w_key] = np.random.normal(0, weight_scale, (lind1, lind))
            self.params[b_key] = np.zeros(layer_sizes[idx])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Start with the input
        tmp = X
        cache_list = []
        if self.normalization == 'batchnorm':
          for indx in range(1, self.num_layers):
            wi=f"W{indx}"
            bi=f"b{indx}"
            gi=f"gamma{indx}"
            bti=f"beta{indx}"
            tmp, cache = affine_bn_relu_forward(tmp, self.params[wi], self.params[bi], self.params[gi], self.params[bti], self.bn_params[indx - 1])
            cache_list.append(cache)

            if self.use_dropout:
              tmp, cache = dropout_forward(tmp, self.dropout_param)
              cache_list.append(cache)

        else:  
          for indx in range(1, self.num_layers):
            wi=f"W{indx}"
            bi=f"b{indx}"
            tmp, cache = affine_relu_forward(tmp, self.params[wi], self.params[bi])
            cache_list.append(cache)

            if self.use_dropout:
              pr=self.dropout_param
              tmp, cache = dropout_forward(tmp, pr)
              cache_list.append(cache)
        wl=f"W{self.num_layers}"
        bl=f"b{self.num_layers}"
        scores, cache = affine_forward(tmp, self.params[wl], self.params[bl])
        cache_list.append(cache)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute regularization term and softmax loss
        reg_term = 0
        for idl in range(1, self.num_layers + 1):
           reg_term += np.sum(self.params[f"W{idl}"] ** 2)
        
        loss, dout = softmax_loss(scores, y)
        mul=0.5 * self.reg * reg_term
        loss +=mul
        #create grads
        grads = {}
        #final layer
        dout, dw, db = affine_backward(dout, cache_list.pop())
        lay=self.num_layers
        grads[f"W{lay}"] = dw + self.reg * self.params[f"W{lay}"]
        grads[f"b{lay}"] = db
        
        ## Backprop through hidden layers with optional dropout and batchnorm, apply L2 regularization
        for layer in reversed(range(1, self.num_layers)):
            if self.use_dropout:
                dout = dropout_backward(dout, cache_list.pop())
        
            if self.normalization == "batchnorm":
                dout, dw, db, gm, db = affine_bn_relu_backward(dout, cache_list.pop())
                grads[f"gamma{layer}"] = gm
                grads[f"beta{layer}"] = db
            else:
                dout, dw, db = affine_relu_backward(dout, cache_list.pop())
            sm=self.reg * self.params[f"W{layer}"]
            grads[f"W{layer}"] = dw + sm
            grads[f"b{layer}"] = db


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
