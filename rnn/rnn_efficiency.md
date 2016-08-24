There's something magical about Recurrent Neural Networks (RNNs).  
Recurren Neural Networks (RNNs)의 마법이 있다.   
I still remember when I trained my first recurrent network for [Image Captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/). 
[Image Captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/)의 RNN을 훈련했을 때가 기억난다.  
Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense.    
몇분의 훈련시간안에,  나의 첫번째 베이비 모델(다소 임의적으로 선택된 하이퍼파라메터를 가진)은 이미지의 매우 좋아 보이는 설명을 생성했다.    

Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your  expectations, and this was one of those times.   
때때로,    

What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I've in fact reached the opposite conclusion).    
이 시점 이 결과를 충격적으로 만드는 것은 RNNs은 본래 훈련하기 어렵다는 것이다(조금 경험후, 반대의 결론에 이르렀다)

Fast forward about a year: I'm training RNNs all the time and I've witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me.   
대략 1년이 지나서, 나는 줄곧 RNN을 훈련시키고 있고, RNN의 힘과 강인함을 목격했다. 그리고 RNN의 마법같은 결과물은 아직도 나를 놀라키는 방법을 찾는다.   

This post is about sharing some of that magic with you.   
이 포스트는 그 마법의 일부를 당신과 공유하는 것에 대한 것입니다.

> We'll train RNNs to generate text character by character and ponder the question "how is that even possible?"
> 텍스트를 한문자씩 생성하는 RNN을 훈련할 것이다. 그리고 질문을 던진다. "그런 것도 가능해?"

By the way, together with this post I am also releasing [code on Github](https://github.com/karpathy/char-rnn) that allows you to train character-level language models based on multi-layer LSTMs.    
그런데, 이 포스트와 함께, [code on Github](https://github.com/karpathy/char-rnn) 도 릴리징했다. 그 코드는 multi-layer LSTM 기반한 캐릭터-단위 언어 모델을 학습한다.

You give it a large chunk of text and it will learn to generate text like it one character at a time.    
RNN에 많은 텍스트 데이터를 주면, 한 문자씩 텍스트를 생성하기 위해 학습한다.   

You can also use it to reproduce my experiments below. But we're getting ahead of ourselves; What are RNNs anyway?

