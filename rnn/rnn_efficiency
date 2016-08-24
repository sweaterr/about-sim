There's something magical about Recurrent Neural Networks (RNNs).  
Recurren Neural Networks (RNNs)의 마법이 있다.   
I still remember when I trained my first recurrent network for [Image Captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/). 
[Image Captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/)의 RNN을 훈련했을 때가 기억난다.  
Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense.    
몇분의 훈련시간안에,  나의 첫번째 베이비 모델(다소 임의적으로 선택된 하이퍼파라메터를 가진)은 이미지의 매우 좋아 보이는 설명을 생성했다.    

Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times.
때때로, 

 What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I've in fact reached the opposite conclusion). Fast forward about a year: I'm training RNNs all the time and I've witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you.

> We'll train RNNs to generate text character by character and ponder the question "how is that even possible?"

By the way, together with this post I am also releasing [code on Github](https://github.com/karpathy/char-rnn) that allows you to train character-level language models based on multi-layer LSTMs. You give it a large chunk of text and it will learn to generate text like it one character at a time. You can also use it to reproduce my experiments below. But we're getting ahead of ourselves; What are RNNs anyway?

