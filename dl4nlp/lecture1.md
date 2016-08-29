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
```
"The cat jumped over the puddle."
```
A good language model will give this sentence a high probability because this is a completely valid sentence, syntactically and semantically. 

Similarly, the sentence "stock boil fish is toy" should have a very low probability because it makes no sense. 

Mathematically, we can call this probability on any given sequence of n words:
![image](https://cloud.githubusercontent.com/assets/1518919/18032646/121aa830-6d46-11e6-8d46-3a62bb907bb5.png)
We can take the unary language model approach and break apart this probability by assuming the word occurrences are completely independent:
![image](https://cloud.githubusercontent.com/assets/1518919/18032645/0d72a576-6d46-11e6-8d1b-e7167e3d69a2.png)
However, we know this is a bit ludicrous because we know the next word is highly contingent upon the previous sequence of words. 

And the silly sentence example might actually score highly. 

So perhaps we let the probability of the sequence depend on the pairwise probability of a word in the sequence and the word next to it. 

We call this the bigram model and represent it as:
![image](https://cloud.githubusercontent.com/assets/1518919/18032647/250497bc-6d46-11e6-9af2-c807da974d8b.png)

Again this is certainly a bit naive since we are only concerning ourselves with pairs of neighboring words rather than evaluating a whole sentence, but as we will see, this representation gets us pretty far along.

 Note in the Word-Word Matrix with a context of size 1, we basically can learn these pairwise probabilities. 

But again, this would require computing and storing global information about a massive dataset.

Now that we understand how we can think about a sequence of tokens having a probability, let us observe some example models that could learn these probabilities.
 
### Continuous Bag of Words Model (CBOW)
One approach is to treat {"The", "cat", ’over", "the’, "puddle"} as a context and from these words, be able to predict or generate the center word "jumped". 

This type of model we call a Continuous Bag of Words (CBOW) Model.

Let’s discuss the CBOW Model above in greater detail. 

First, we set up our known parameters. 

Let the known parameters in our model be the sentence represented by one-hot word vectors. 

The input one hot vectors or context we will represent with an $x^{(c)}$. 

And the output as $y^{(c)}$ and in the CBOW model, since we only have one output, so we just call this $y$ which is the one hot vector of the known center word. 

Now let’s define our unknowns in our model.

We create two matrices, $\mathcal{V} ∈ R^{n×|V|}$ and $\mathcal{U} ∈ R^{|V|×n}$ . Where n is an arbitrary size which defines the size of our embedding space. 

$\mathcal{V}$ is the input word matrix such that the $i$-th column of $\mathcal{V}$ is the n dimensional embedded vector for word $w_i$ when it is an input to this model.

We denote this $n × 1$ vector as $v_i$. 

Similarly, $\mathcal{U}$ is the output word matrix. 

The $j$-th row of $\mathcal{U}$ is an n-dimensional embedded vector for word $w_j$ when it is an output of the model. 

We denote this row of $\mathcal{U}$ as $u_j$. 

Note that we do in fact learn two vectors for every word $w_i$(i.e. input word vector $v_i$ and output word vector $u_i$)

We breakdown the way this model works in these steps:

1. We generate our one hot word vectors $(x^{(c−m)}, . . . , x^{(c−1)}, x^{(c+1)}, . . . , x^{(c+m)})$ for the input context of size m
2.  We get our embedded word vectors for the context $(v^{c−m} =\mathcal{V}x^{(c−m)}, v^{c−m+1} = \mathcal{V}x^{(c−m+1)}, . . ., v^{c+m} = \mathcal{V}x^{(c+m)})$.
3. Average these vectors to get ![image](https://cloud.githubusercontent.com/assets/1518919/18033406/6506aa90-6d5e-11e6-8cee-96549bd8a8bf.png)
4. Generate a score vector $z = \mathcal{U}\hat{v}$
5. Turn the scores into probabilities $\hat{y} = softmax(z)$
6. We desire our probabilities generated, $\hat{y}$, to match the true probabilities, $y$, which also happens to be the one hot vector of the actual word. 

So now that we have an understanding of how our model would work if we had a $\mathcal{V}$ and $\mathcal{U}$, how would we learn these two matrices? 

Well, we need to create an objective function. 

Very often when we are trying to learn a probability from some true probability, we look to information theory to give us a measure of the distance between two distributions. 

Here, we use a popular choice of distance/loss measure, cross entropy $H(\hat{y}, y)$.

The intuition for the use of cross-entropy in the discrete case can be derived from the formulation of the loss function:
![image](https://cloud.githubusercontent.com/assets/1518919/18033427/03e91d28-6d5f-11e6-9225-220af452a64b.png)
Let us concern ourselves with the case at hand, which is that $y$ is a one-hot vector. 

Thus we know that the above loss simplifies to simply:
![image](https://cloud.githubusercontent.com/assets/1518919/18033434/1b020c4a-6d5f-11e6-83de-ae5319ec061b.png)

In this formulation, $c$ is the index where the correct word’s one hot vector is 1. 

We can now consider the case where our prediction was perfect and thus $\hat{y}_c = 1$. 

We can then calculate $H(\hat{y}, y) = −1\log(1) = 0$. 

Thus, for a perfect prediction, we face no penalty or loss. 

Now let us consider the opposite case where our prediction was very bad and thus $ \hat{y}_c= 0.01. $

As before, we can calculate our loss to be $H(\hat{y}, y) = −1 \log(0.01) ≈ 4.605$. 

We can thus see that for probability distributions, cross entropy provides us with a good measure of distance. 

We thus formulate our optimization objective as:

![image](https://cloud.githubusercontent.com/assets/1518919/18033513/572ecab6-6d62-11e6-86d5-ae6fcb12e1e2.png)

We use gradient descent to update all relevant word vectors $u_c$ and $v_j$

### 4.3 Skip-Gram Model

Another approach is to create a model such that given the center word "jumped", the model will be able to predict or generate the surrounding words "The", "cat", "over", "the", "puddle". 

Here we call the word "jumped" the context. 

We call this type of model a SkipGram model. 

Let’s discuss the Skip-Gram model above. 

The setup is largely the same but we essentially swap our $x$ and $y$ i.e. $x$ in the CBOW are now $y$ and vice-versa.

The input one hot vector (center word) we will represent with an $x$ (since there is only one). 

And the output vectors as $y^{(j)}$. 

We define $\mathcal{V}$ and $\mathcal{U}$ the same as in CBOW.

![image](https://cloud.githubusercontent.com/assets/1518919/18033630/4edc0b00-6d65-11e6-8a0c-fe6791bb8830.png)

As in CBOW, we need to generate an objective function for us to evaluate the model. 

A key difference here is that we invoke a Naive Bayes assumption to break out the probabilities. 

If you have not seen this before, then simply put, it is a strong (naive) conditional independence assumption.

 In other words, given the center word, all output words are completely independent.
.![image](https://cloud.githubusercontent.com/assets/1518919/18033673/08547658-6d66-11e6-8c38-b4ed453c92f4.png)

With this objective function, we can compute the gradients with respect to the unknown parameters and at each iteration update them via Stochastic Gradient Descent.

### 4.4 Negative Sampling

Lets take a second to look at the objective function. 

Note that the summation over $|V|$ is computationally huge! 

Any update we do or evaluation of the objective function would take $O(|V|)$ time which if we recall is in the millions. 

A simple idea is we could instead just approximate it.

For every training step, instead of looping over the entire vocabulary, we can just sample several negative examples! 

We "sample" from a noise distribution ($P_n(w)$) whose probabilities match the ordering of the frequency of the vocabulary. 

To augment our formulation of the problem to incorporate Negative Sampling, all we need to do is update the:

• objective function
• gradients
• update rules

Mikolov et al. present Negative Sampling in Distributed Representations of Words and Phrases and their Compositionality. 

While negative sampling is based on the Skip-Gram model, it is in fact optimizing a different objective.

Consider a pair $(w, c)$ of word and context. 

Did this pair come from the training data? 

Let’s denote by $P(D = 1|w, c)$ the probability that $(w, c)$ came from the corpus data.

Correspondingly, $P(D = 0|w, c)$ will be the probability that (w, c) did not come from the corpus data. 

First, let’s model $P(D = 1|w, c)$ with the sigmoid function:
![image](https://cloud.githubusercontent.com/assets/1518919/18033875/4e30e64a-6d69-11e6-8295-13fa1f4c6dc6.png)

Now, we build a new objective function that tries to maximize the probability of a word and context being in the corpus data if it indeed is, and maximize the probability of a word and context not being in the corpus data if it indeed is not. 

We take a simple maximum likelihood approach of these two probabilities. 

(Here we take $\theta$ to be the parameters of the model, and in our case it is $\mathcal{V}$ and $\mathcal{U}$.)

![image](https://cloud.githubusercontent.com/assets/1518919/18033925/9d070d3e-6d6a-11e6-94d1-f46a06891886.png)

Note that $\tilde {D} $ is a "false" or "negative" corpus. 

Where we would have sentences like "stock boil fish is toy". 

Unnatural sentences that should get a low probability of ever occurring. 

We can generate $\tilde {D} $  on the fly by randomly sampling this negative from the word bank. 

Our new objective function would then be:
![image](https://cloud.githubusercontent.com/assets/1518919/18033997/7f1ee470-6d6c-11e6-8b76-6d1ee2e72663.png)
In the above formulation,$\{\tilde{u}_k|k = 1 . . . K\}$ are sampled from $P_n(w)$.

Let’s discuss what $P_n(w)$ should be. 

While there is much discussion of what makes the best approximation, what seems to work best is the Unigram Model raised to the power of 3/4. 

Why 3/4? 

Here’s an example that might help gain some intuition:






