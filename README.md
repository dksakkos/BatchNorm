# BatchNorm
A re-implenetation of the BatchNormalization layer in Keras

[Original paper](https://arxiv.org/pdf/1502.03167.pdf)

# Main motivation of the paper
A deep neural network is usually trained with the back-propagation algorithm. Specifically, a loss function is used to calculate the error between the network's prediction and the ground truth. Then, the gradient is computed with respect to the network's parameters. Once this is done, the weights of each layer of the network are adjusted accordingly.
Considering this is done for all layers, one can easily realize that as the weights are changed during training, so does their distribution. Compounded by the fact that each layer receives the previous layer's output as input, it is evident that early, small changes will be amplified in deeper layers, leading to great imbalances of the layer's features in terms of scale. This is of course exarcebated if the number of layers in a network increase, making the training of very deep networks slow at best, and nearly impossible at worst.


# A description of the proposed block and an enumeration of its operations
The authors propose a layer that restores and equalizes the distribution and scale of the features, for all layers in the network. This is done by normalizing the features in a mini-batch to have zero mean and unit standard deviation. However, these values might not be optimal and can constrain the representational ability of the model, so it is best to let the network learn them instead. This is done using two variables, gamma and beta, which are learned with back-propagation along with the rest of the network's parameters.
Of course, computing the statistics of the whole dataset instead of a mini-batch would be preferable, but is in most cases infeasible. However, it is a good approximation when the batch size is large enough. That said, it is necessary our network gives deterministic predictions during inference, regardless of the batch. For this reason, the normalization is done using moving averages when the network is in inference mode.


Here is a list of the proposed layer's operations:
1. Calculate mini-batch mean 
1. Calculate mini-batch standard deviation
1. Normalize the features by subtracting the mean and dividing by the standard deviation
1. Re-scale the features by multiplying by gamma and adding beta

![Training accuracy](/Figures/acc.png?raw=true)

![Validation accuracy](/Figures/val_acc.png?raw=true)

![Training loss](/Figures/loss.png?raw=true)

![Validation loss](/Figures/val_loss.png?raw=true)

As seen from the plots above, the model with no batch normalization converged slower than the other models, and was outperformed by a significant margin in terms of accuracy. Therefore, the results from the original paper have been replicated successfully in this experiment.

Moreover, it can be observed that the model trained with the custom implementation of Batch Normalization follows a very similar loss and accuracy curve, when compared to the official implementation. This indicates that our implementation is very close to that of TensorFlow.
