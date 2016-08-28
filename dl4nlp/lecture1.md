http://cs224d.stanford.edu/lecture_notes/notes1.pdf

## Iteration Based Methods

Let us step back and try a new approach. 

Instead of computing and storing global information about some huge dataset (which might be billions of sentences), we can try to create a model that will be able to learn one iteration at a time and eventually be able to encode the
probability of a word given its context.

We can set up this probabilistic model of known and unknown parameters and take one training example at a time in order to learn just a little bit of information for the unknown parameters based on the input, the output of the model, and the desired output of the model.

At every iteration we run our model, evaluate the errors, and follow an update rule that has some notion of penalizing the model parameters that caused the error. 

This idea is a very old one dating back to 1986. 

We call this method "backpropagating" the errors (see Learning representations by back-propagating errors. David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1988).)

### 4.1 Language Models (Unigrams, Bigrams, etc.)
First, we need to create such a model that will assign a probability to
a sequence of tokens. Let us start with an example: