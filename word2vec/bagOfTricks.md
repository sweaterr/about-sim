---
layout: post
title: Bag of Tricks for Efficient Text Classification
---
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

More precisely, the computational complexity is $O(kh)$ where $k$ is the number of classes and $h$ the dimension of the text representation. 

In order to improve our running time, we use a hierarchical softmax (Goodman, 2001) based on the Huffman coding tree (Mikolov et al., 2013). 

During training, the computational complexity drops to $O(h \log_2(k))$.

The hierarchical softmax is also advantageous at test time when searching for the most likely class. 

Each node is associated with a probability that is the probability of the path from the root to that node. 

If the node is at depth $l + 1$ with parents $n_1, . . . , n_l$ , its probability is
![image](https://cloud.githubusercontent.com/assets/1518919/18032004/d95f63fe-6d32-11e6-8066-dac8e4fe0279.png)

This means that the probability of a node is always lower than the one of its parent. 

Exploring the tree with a depth first search and tracking the maximum
probability among the leaves allows us to discard any branch associated with a small probability. 

In practice, we observe a reduction of the complexity to $O(h \log_2(k))$ at test time. 

This approach is further extended to compute the $T$-top targets at the cost of $O(log(T))$, using a binary heap.

#### 2.2 N-gram features
Bag of words is invariant to word order but taking explicitly this order into account is often computationally very expensive. 

Instead, we use a bag of n-grams as additional features to capture some partial information about the local word order. 

This is very efficient in practice while achieving comparable results to methods that explicitly use the order (Wang and Manning, 2012).

We maintain a fast and memory efficient mapping of the n-grams by using the hashing trick (Weinberger et al., 2009) with the same hashing function as in Mikolov et al. (2011) and 10M bins if we only used bigrams, and 100M otherwise.

