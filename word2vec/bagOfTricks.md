## Bag of Tricks for Efficient Text Classification
https://arxiv.org/pdf/1607.01759v3.pdf

### Model architecture

![image](https://cloud.githubusercontent.com/assets/1518919/18031769/f3b1bb84-6d29-11e6-9334-bab5225d39b6.png)
Figure 1 shows a simple linear model with rank constraint. 

The first weight matrix A is a look-up table over the words. 

The word representations are then averaged into a text representation, which is in turn fed to a linear classifier. 

The text representation is an hidden variable which can be potentially be reused. 

This architecture is similar to the cbow model of Mikolov et al. (2013), where the middle word is replaced by a label. 

We use the softmax function $f$ to compute the probability distribution over the predefined classes. 

For a set of $N$ documents, this leads to minimizing the negative loglikelihood over the classes:

![image](https://cloud.githubusercontent.com/assets/1518919/18031924/96282f06-6d2f-11e6-9142-aecf2f56999f.png)
where $x_n$ is the normalized bag of features of the $n$-th document, $y_n$ the label, A and B the weight matrices.

This model is trained asynchronously on multiple CPUs using stochastic gradient descent and a linearly decaying learning rate.

#### 2.1 Hierarchical softmax
When the number of classes is large, computing the linear classifier is computationally expensive. 

More precisely, the computational complexity is O(kh) where k is the number of classes and h the dimension of the text representation. 

In order to improve our running time, we use a hierarchical softmax (Goodman, 2001) based on the Huffman coding tree (Mikolov et al., 2013). During training, the computational complexity drops to O(h log2
(k)).