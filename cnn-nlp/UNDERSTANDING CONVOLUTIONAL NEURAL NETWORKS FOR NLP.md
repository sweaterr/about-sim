다음 [포스트](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)를, 좀 더 자세히 읽으려는 공부목적으로 번역해보았습니다. 

Convolutional Neural Network (CNNs)에 대해서 들었을 때, 일반적으로 컴퓨터 비전을 생각한다. CNNs은 페이스북의 자동 포토 태깅부터 자율 주행차까지 이미지 분류와 대부분의 컴퓨터 비전 시스템의 혁명을 책임 졌다. 

최근엔, CNNs이 자연어처리에 적용되기 시작해서 재밌는 결과를 얻고 있다. 이 포스트에서는 CNN이 NLP에서 어떻게 쓰이고 있는지 요약할 것이다. CNNs에 숨겨진 직관은 컴퓨터 비전 use case에서는 이해하기가 쉽다. 그래서 거기서부터 시작하고 천천히 NLP로 넘어가겠다.

### WHAT IS CONVOLUTION?
나에게 있어서, 합성곱(convolution)를 이해하는 가장 쉬운 방법은 행렬에 적용되는 슬라이딩 윈도우 함수로 생각하는 것이다. 말로하면 어렵지만, 시각화를 해보면 매우 이해가 쉽다.

![enter image description here](http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif)

왼쪽에 있는 행렬이 흑백 이미지를 표현하고 있다고 상상해봐라. 각 entry는 하나의 픽셀에 대응한다. 0은 블랙 1은 화이트 (일반적으로 grayscale 이미지는 0에서 255사이이다)  슬라이딩 윈도우는 커널, 필터, 특성 검출기라고도 불린다. 여기서는 3×3 필터를 사용하고, 각 원본 행렬에 element-wise로 값을 곱한 후 합한다. 전체 합성곱을 얻기 위해, 전체 행렬에 대해서 필터를 미끄러뜨리면서 이 일을 한다.

왜 이 일을 해야하는지 궁금할 것이다. 여기 이해하기 쉬운 예제가 있다.

### 각 픽셀을 주변 픽셀값으로 평균취하는 것은 이미지를 흐리게 한다.

![enter image description here](http://docs.gimp.org/en/images/filters/examples/convolution-blur.png)
![enter image description here](http://docs.gimp.org/en/images/filters/examples/generic-taj-convmatrix-blur.jpg)

### 픽셀과 주변 사이의 차이는 가장자리를 찾는다.
(이것을 직관적으로 이해하기 위해, 각 픽셀을 주변 픽셀과 같은 스무딩된 이미지 부분에서 어떤일이 일어나는지 생각해보자:  더하기는 취소되고 결과값은 0 즉 블랙이 된다. If there’s a sharp edge in intensity, a transition from white to black for example, you get a large difference and a resulting white value)
![enter image description here](http://docs.gimp.org/en/images/filters/examples/generic-taj-convmatrix-edge-detect.jpg)
[GIMP](http://docs.gimp.org/en/plug-in-convmatrix.html) 메뉴얼에 다른 예제들이 있다.

To understand more about how convolutions work I also recommend checking out Chris Olah’s post on the topic.
합성곱이 어떻게 동작하는지 좀 더 이해하기 위해서  [ ](http://colah.github.io/posts/2014-07-Understanding-Convolutions/) 를 참고하기를 추천한다.

### WHAT ARE CONVOLUTIONAL NEURAL NETWORKS?
이제 합성곱은 알았다 , CNNs은 어떤가? CNNs은 기본적으로  몇개의 convolutions 층을 가지고, 층의 출력에 비선형 활성화 함수(ReLU or tanh)를 적용한다. 전통적 피드 포워드 네트워크에서는 각 입력 뉴런을 다름 층의 출력 뉴런에 연결한다. 그것은 완전 연결 층 또는 아핀 층이라고 불린다. CNNs에서는 그것을 하지 않는다. 대신,  출력을 계산하기 위해, 입력층에 합성곱을 사용한다. 이는  입력의 각 지역은 출력에 있는 뉴런에 연결된 지역연결을 야기한다. 각 층은 일반적으로 몇백 또는 몇천 개의 다른 필터를 적용하는데, 위에서 봤듯이, 그 결과는 합쳐진다. 또한 pooling(subsampling) 층이라고 불리는 어떤 것이 있다. 그러나 이는 나중에 다룰 것이다. 학습시간에는, CNN는 수행하길 원하는 작업에 기반한 **필터의 값을 자동적으로 학습**한다. 예를 들어, 이미지 분류에서, CNN은 첫번째 층에서  원본 픽셀에서 가장자리를 찾는 것을 학습할 수 있다. 그 후 두번째 층에서 그 가장자리를 사용해서 간단한 모양을 찾아낸다. 그 후 좀 더 높은 수준의 피쳐를 찾기 위해 이 모양을 이용한다. 예를 들어 얼굴 모양 같은 것이 그것이다. 마지막 층에서는, 이런 고수준 특성을 사용해서 분류한다.

![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-07-at-7.26.20-AM.png)

이 계산에는 주목해야 할 두가지 점이 있다: **Location Invariance** 와 **Compositionality**. 이미지에 코끼리가 있는지 없는지 분류하고 싶다고 하자. 전체 이미지에 대해서 필터를 슬라이딩시키고 있기 때문에, 코끼리가 어디에 있는지 신경쓰지 않아도 된다. 실제로, pooling은 변환, 회전, 스케일링에 대한 불변성을 부여한다. 하지만 이건 나중에 다룰 것이다. 두번째 주요 특성은 (지역) compositionality이다. 각 필터는 저수준 특성의 지역 조각을 고수준 표현으로 합성한다. CNN이 컴퓨터 비전에서 파워풀한 이유이다. 픽셀로부터 가장자리를 만들고, 가장자리로부터 모양을 만들고 모양으로부터 복잡한 물체를 만든다는 것은 직관적이다.

### 그래서, 어떻게 이걸 NLP에 적용하죠?
이미지 픽셀대신, 대부분 NLP 문제의 입력은 행렬로 표현되는 문장이나 문서이다. 행렬의 각 열은 토큰, 일반적으로 단어에 대응한다. 그러나, 문자일 수도 있다. 즉, 각 행은 단어를 표현하는 벡터이다. 일반적으로, 이런 벡터는 word2vec 도는 GloVe같은 단어 임베딩(저차원 표현)이다. 그러나 단어집 안의 단어를 인덱스화한 one-hot 벡터일수도 있다. 10 단어 문장에 대해서, 100차원 임베딩을 사용하여, 10×100 행렬을 입력으로 할 수 있다. 그것의 우리의 "이미지"이다. 

비전에서,  필터는 이미지의 지역 조각을 슬라이딩한다. 하지만, NLP에서는 일반적으로 행렬의 전체 행(단어)을 슬라이딩한다. 그러므로, 필터의 너비는 보통 입력 행렬의 너비와 같다. 높이 또는 지역 크기는 변한다. 그러나 슬라이등 윈도우는 2-5단어가 일반적이다. 모두 합치면, NLP에서의 CNN은 다음처럼 보인다. (몇 분의 시간을 내서, 그림을 이해해 보자. 어떻게 차원이 계산되는가? 지금은 pooling 은 무시해라. 나중에 설명한다.)
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png)

문장분류에 대한 CNN 구조. 여기서,  세개의 필터 지역 크기는 2, 3, 4이다. 각각은 두 개의 필터를 가지고 있다. 모든 필터는 문장 행렬에 대해서 합성곱을 수행하고, (가변-길이) feature map를 생성한다. 그후 1-max pooling이 전체 map에 대해서 수행된다 . 즉 각 feature map에서 가장 큰 수가 기록된다. 그러므로, univariate feature vector는 모든 6개의 map에서 부터 만들어지고, 이 6개 feature들은 penultimate layer에 대한 특성 벡터를 형성하기 위해 이어붙여진다. 마지막 softmax 층은 이 특성 벡터를 입력으로 받아서 문장을 분류하는데 사용한다; 여기서 이진 분류를 가정하므로 두 개의 가능한 출력상태가 있다.
Source: Zhang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification.

컴퓨터 비전에서 얻은 그 좋은 직관은 어떻게 된건가? Location Invariance 과 local Compositionality는 이미지에 대한 직관적인 감각을 만들었지만, NLP에서는 잘 통하지 않는다. 너는 문장에서 단어가 어디에서 등장하는지가 매우 신경쓸수 있다. 서로 가까운 픽셀은 의미론적으로 연관이 있을 가능성이 있다. 그러나, 단어들은 때론 그렇지 않다.
In many languages, parts of phrases could be separated by several other words. 
많은 언어에서, 절은  몇몇 다른 단어로 분리 될 수 있다.
The compositional aspect isn’t obvious either. 
compositional 면도 역시 명백하지 않다.
Clearly, words compose in some ways, like an adjective modifying a noun, but how exactly this works what higher level representations actually “mean” isn’t as obvious as in the Computer Vision case.
명백히, 단어는 여러 방법으로 조합된다.  

이렇게 봤을 때, CNNs는 NLP 문제에 잘 맞지 않는것처럼 보인다. [Recurrent Neural Networks](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)이 좀더 직관적으로 합리적으로 보인다. 그것은  우리가 언어를 처리하는 방식과 닮아 있다: 왼쪽에서 오른쪽으로 읽는것처럼. 운좋게도, 이는 CNNs이 동작하지 않는다는 것을 의미하진 않는다. [모든 모델은 틀렸다. 하지만 어떤 것은 쓸모있다.](https://en.wikipedia.org/wiki/All_models_are_wrong) CNN를 NLP에 적용하는것은 매우 잘 동작한다는 것이 밝혀졌다. 간단한 Bag of Words model은 부적절한 가정의 과도한 간소화이다. 그러나 오랜기간 표준 접근법이었으면, 좋은 성능을 낸다.


CNN의 가장 큰 장점은 매우, 매우 빠르다는 것이다. 합성곱은 컴퓨터 그래픽스의 핵심이며, GPU레벨로 구현되어 있다. [n-gram](https://en.wikipedia.org/wiki/N-gram)같은 것과 비교하면, CNN은 또한 표현의 측면에서 매우 효율적이다. 큰 단어집을 가지고, 3-gram 이상의 어떤것을 계산하는 것은 매우 계산이 비싸질 수 있다. 심지어 구글도 5-gram이상의 어떤것도 하지 못한다. Convolutional Filters는 전체 단어집을 표현할 필요없이, 자동적으로 좋은 표현을 학습한다. 5이상의 크기를 가진 필터를 쓰는 것도 완전 가능하다. 첫번째 층에서 많은 학습된 필터가 n-gram 과 매우 비슷한 특성을 잡아내고, 좀더 compact한 방법으로 그것을 표현한다.

### CNN HYPERPARAMETERS
CNN을 어떻게 NLP에 적용하는지 설명하기 전에,  CNN를 만들때 해야할 몇가지 선택을 보자. 이것이 필드에 대한 더 깊은 이해가 되기를 바란다.

### NARROW VS. WIDE CONVOLUTION
convolutions를 설명할 때, 필터를 어떻게 적용하는지에 대한 세부사항을 등한시했다. 3×3 filter를 행렬 가운데에 적용하는 것은 잘 동작한다. 하지만 가장가리에 적용한다면 어떻게 되는가? 위와 왼쪽에 이웃원소가 없는 행렬의 첫번째 원소에 필터를 어떻게 적용할수 있을까? zero-padding을 사용할 수 있다. 행렬 밖에 있는 모든 원소는 모두 0으로 채운다. 이렇게 함으로써, 필터를 입력 행렬의 모든 원소에 적용할 수 있고, 크거나 동일한 크기의 출력을 얻는다. zero-padding를 추가하는 것은 소위 wide convolution라고 불린다.  zero-padding을 사용하지 않는 것을 narrow convolution이라 부른다. 1차원에서의 예제는 다음과 같다. 
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-05-at-9.47.41-AM.png)
wide convolution이 어떻게 유용한지 또는 심지어 필수 인지는 입력 크기에 비해서 큰 필터를 적용할 때 볼수 있다. 위의 예에서, narrow convolution는 `(7-5)+1=3`의 출력을 내고,  wide convolution는 `(7+2*4-5)+1=11`의 출력을 낸다. 좀 더 일반적으로, 출력 크기에 대한 공식은 $n_{out}=(n_{in} + 2*n_{padding} - n_{filter}) + 1$

### 보폭(STRIDE SIZE)
convolutions에 대한 다른 hyperparameter는 보폭(stride size()이다. 보폭는 한번에 필터를 얼마나 움직일것이냐에 대한 크기이다. 위의 예에서는 모두 보폭이 1이고, 필터의 연속 적용은 overlap된다. 큰 보폭은 필터의 적용의 횟수를 낮추고 때문에 크기가 작은 출력을 만든다. [Stanford cs231 website](http://cs231n.github.io/convolutional-networks/) 에서의 다음 예는 보폭 1, 2가 1차원 입력에 적용됐을 때 모습을 보여준다.
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-05-at-10.18.08-AM.png)
문헌에서는 보통 보폭 1을 볼 수 있다. 그러나, 큰 보폭 사이즈는  Recursive Neural Network와 다소 비슷한 행동을 하는 모델을 만들 수 있다. 즉 트리와 같은.

### POOLING LAYERS
Convolutional Neural Networks의 주요 특징은 일반적으로 convolutional 층 다음에 적용되는 pooling 층이다.  Pooling 층은 입력을 subsample한다. 가장 흔한 pooling 방법은 각 필터의 결과에 max 연산을 적용하는 것이다. 전체 행렬에 대해서 pool할 필요는 없고, 윈도우 단위로 할 수 있다. 예를 들어, 다음은 2×2 윈도우에 해한 max pooling을 보여준다 (NLP에서, 일반적으로 전체 출력에 대한 pooling을 적용한다. 즉 각 필터에서 하나의 숫자만 나온다):
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-05-at-2.18.38-PM.png)
왜 풀링을 하는가? 몇가지 이유가 있다. 풀링의 한가지 성질은 일반적으로 분류에 쓰이는 고정된 크기의 출력 행렬를 제공한다는 것이다. 예를 들어, 1,000개의 필터를 가지고 있고, 각각에 대해서 맥스 풀링을 적용했다면, 필터의 크기나 입력의 크기와 상관없이, 1000차원 출력을 얻게된다, 이는 가변 크기의 문장, 가변 크기의 필터를 사용하지만, 분류기에 넣어줄  항상 같은 크기의 출력 차원을 얻게 해준다. 풀링은 또한 출력 차원을 감소시키지만, (바라건데) 대부분의 알짜 정보는 유지시킨다. 각 필터를 예를 들어 “not amazing”같은 부정문을 문장이 포함하는지 찾는것같은 특정 feature를 추출해주는 것으로 생각할 수 있다.  이 구절이 문장의 어느곳에 등장했다면, 그 지점에 대한 필터 적용의 결과는 큰 값을 줄 것이고, 나머지 지점에서는 작은 값을 줄것이다. max연산을 수행함으로써, 특성이 문장에서 나오는지 안나오는지에 대한 정보를 유지할수 있다. 하지만 어디서 그 것이 나타나는지에 대한 정보는 잃는다. 그러나 이 정보가 정말 유용한가? 그렇다. 사실 bag of n-grams model이 하는 일과 비슷하다. locality(문장에서 어디에서 일어나는지)에 대한 global 정보는 잃지만, 필터가 잡은  “amazing not”는 “not amazing”와 매우다르다 같은  local 정보는 유지할 수있다. 이미지 인식에서는 풀링은 translating(shifting)과 회전에 대한 기본적인 invariance을 제공한다. 어떤 지역을 풀링할때, 출력은 이미지를 shift/rotate가 되고 거의 같다. 왜냐하면 max 연산은 그와 상관없이 같은 값을 고를것이기 때문이다.

### CHANNELS
이해할 필요가 있는 마지막 개념은 채널이다. 채널은 입력의 다른 “views”이다. 예를 들어, 이미지 인식에서, RGB (red, green, blue) 채널을 일반적으로 사용한다. 다른 또는 같은 가중치를 가지고 convolutions을 여러 채널에 적용할 수 있다.  NLP에서도 물론 다양한 채널을 생각할 수 있다: 다른 단어 임베딩(예를 들어, word2vec 와 GloVe)에 대해서 분리된 채널을 쓸수 있고, 다른 언어로 표현된 같은 문장에 대해서 하나의 채널을 사용할 수도 있다.

### CONVOLUTIONAL NEURAL NETWORKS APPLIED TO NLP
NLP에 대한 CNN의 응용을 살펴보자.연구 결과 몇개를 요약할 것이다. 불가피하게, 많은 재밌는 응용을 놓치게 될것이다(답변으로 알려달라). 그러나 최소한 인기있는 결과 몇개를 다루길 원한다.  가장 자연스러운 CNN의 응용은 Sentiment Analysis, Spam Detection, Topic Categorization같은 분류 문제이다. Convolutions 와 pooling 연산은 단어의 지역 순서에 대한 정보를 잃는다. 그래서 PoS 태깅 또는 개체명 추출같은 순서 태깅은 순수한 CNN 구조를 사용하기엔 좀 어렵다(불가능하지는 않지만, 입력에 positional 특성을 추가해야 한다)
 
[1]은 CNN 구조를 Sentiment Analysis 와 Topic Categorization task같은 다양한 분류 데이터셋에 대해서 평가 했다.  CNN 구조는 모든 데이터에 대해서 매우 좋은 성능을 보였고, 몇몇은 최고의 성능을 보인다. 놀랍게도, 이 논문에서 쓰인 네트워크는 매우 간단하다. 그래서 더 파워풀하다. 입력층은 word2vec 단어 임베딩을 연결한 문장이다. 그 후에, multiple filters를 가진 convolutional 층, max-pooling 층을 연결하고,  마지막엔 softmax 분류기가 된다. 논문은 또한 static 과 dynamic word embeddings 형태의 두 개의 다른 채널을 실험했다. 하나의 채널은 학습동안에 조정되지만 다른것은 조정되지 않는다. 비슷하게, 다소 복잡한 구조는 [2]에서 제안됐다. [6] 은 “semantic clustering”을 수행하는 추가 층을 네트워크 구조에 추가한다.
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-8.03.47-AM.png)
[4]는 word2vec 또눈 GloVe같은  pre-trained된 단어 벡터 없이, CNN를 밑바닥부터 훈련했다.  convolutions를 one-hot vectors에 바로 적용했다. 저자는 네트워크의 학습할 파라메터를 줄이는 입력 데이터에 대한 공간 효율적인 bag-of-words같은 표현을 제안했다.  [5]에서는 저자는 텍스트 지역의 문맥을 예측하는 CNN을 이용해서 학습하는 추가적인 비지도 “region embedding” 을 가진 모델을 확장했다. 이 논문들에서의  접근법은 long-form texts(영화 리뷰같은)에 대해서 매우 잘 동작하는 것처럼보이지만, short texts(트윗같은)에 대해서는 명확하지 않다.  직관적으로 짧은 텍스트에 대한 pre-trained word embeddings를 사용하는 것은 긴 텍스트에서 사용하는 것보다 이점이 있어 보인다.

CNN 구조를 만드는 것은 선택해야할 많은 hyperparameters가 있다는 것을 의미한다: Input 표현 (word2vec, GloVe, one-hot), convolution 필터의 수와 크기, pooling 전략(max, average), activation 함수(ReLU, tanh). [7]은 CNN에서 hyperparameters를 바꾸는데 대한 empirical한 평가를 했다.  텍스트 분류에 대한 너만의 CNN를 찾고 있다면, 이 논문의 결과를 시작점으로 잡는 것은 좋은 생각이 될것이다. max-pooling는 항상 average pooling을 이긴다. 이상적인 필터 크기는 중요하지만 문제마다 다르다. regularization은 NLP 문제에서 중요한 것으로 보이진 않는다.  이 연구의 위험한 점은 모든 데이터셋이 문서길이 관점에서 매우 비슷하다는 것이다. 그래서 다른 모양의 데이터셋에 대해서 잘 적용될지는 모르겠다.

[8]은 Relation Extraction 와 Relation Classification tasks에 대한 CNN을 탐구했다. word vector에 추가적으로, 저자는 convolutional층에서 단어의 상대적 위치를 입력에서 주요 entities로 사용했다. 이 모델은 entities의 위치가 주어지면, 각 예제 입력은 하나의 관계를 가진다고 가정했다. [9] 와 [10]도 비슷한 모델을 탐구했다.

CNN의 다른 재미있는 use case는  Microsoft Research에서 나온 [11] 와 [12]에서 찾을 수 있다. 이 논문들은 정보검색 분야에서 사용할 수 있는 문장의 의미론적으로 의미있는 표현을 어떻게 학습할 수 있는지 기술했다. 논문에서 나온 예제는 유저에게 그가 읽은 최근 문서를 기반으로 잠재적으로 재미있는 문서를 추천해준다. 문장 표현은 검색엔진의 로그 데이터 기반으로 학습된다.

대부분의 CNN 구조는 학습의 일부분으로 단어와 문장에 대한 embeddings 학습한다.  모든 논문이 임베딩 학에 집중하지는 않는다.  [13] 는 Facebook 포스트의 해쉬태그를 예측하고 동시에 의미있는 문장과 단어의 임베딩을 생성하는 CNN 구조를 소개했다. 이 학습된 임베딩은 다른 문제에 성공적으로 적용됐다 - 클릭 데이터에 기반한 사용자에게 재미있어할만한 문서 추천하기

### CHARACTER-LEVEL CNNS
여태까지는 모두 단어에 기반한 모델을 소개했다. 그러나 문자에 CNN을 바로 적용하는 연구도 있다. [14]는  문자-레벨 임베딩을 학습하고,  pre-trained word embeddings과 조인해서 Speech 태깅을 위한 CNN에 사용한다.  [15][16]는 사전학습된 임베딩 없이, 문자로부터 직접 학습하는 CNN의 사용을 탐구했다. 명백히, 9개층의 상대적으로 깊은 네트워크를 사용했고, Sentiment Analysis 과 Text Categorization에 적용했다.

결과는 문자-레벨 입력으로부터 직접적으로 학습하는 것은 큰 데이터셋(100만)에서 잘 동작함을 보였다. 그러나 작은 데이터셋(10000)에 대해서는 간단한 모델을 못이겼다.

명백히, [Natural Language Processing (almost) from Scratch](http://arxiv.org/abs/1103.0398)에 있는 것처럼  NLP에 CNN을 적용하는 훌륭한 업적은 있었다. 그러나 새로운 결과와 최신 시스템에 대한 연구속도는 가속화되고 있다.

### PAPER REFERENCES
[\[1\] Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 1746–1751.](http://arxiv.org/abs/1408.5882)
[\[2\] Kalchbrenner, N., Grefenstette, E., & Blunsom, P. (2014). A Convolutional Neural Network for Modelling Sentences. Acl, 655–665.](http://arxiv.org/abs/1404.2188)
[\[3\] Santos, C. N. dos, & Gatti, M. (2014). Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts. In COLING-2014 (pp. 69–78).](http://www.aclweb.org/anthology/C14-1008)
[\[4\] Johnson, R., & Zhang, T. (2015). Effective Use of Word Order for Text Categorization with Convolutional Neural Networks. To Appear: NAACL-2015, (2011).](http://arxiv.org/abs/1412.1058v1)
[\[5\] Johnson, R., & Zhang, T. (2015). Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding.](http://arxiv.org/abs/1504.01255)
[\[6\] Wang, P., Xu, J., Xu, B., Liu, C., Zhang, H., Wang, F., & Hao, H. (2015). Semantic Clustering and Convolutional Neural Network for Short Text Categorization. Proceedings ACL 2015, 352–357.](http://www.aclweb.org/anthology/P15-2058)
[\[7\] Zhang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification,](http://arxiv.org/abs/1510.03820)
[\[8\] Nguyen, T. H., & Grishman, R. (2015). Relation Extraction: Perspective from Convolutional Neural Networks. Workshop on Vector Modeling for NLP, 39–48.](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)
[\[9\] Sun, Y., Lin, L., Tang, D., Yang, N., Ji, Z., & Wang, X. (2015). Modeling Mention , Context and Entity with Neural Networks for Entity Disambiguation, (Ijcai), 1333–1339.](http://ijcai.org/papers15/Papers/IJCAI15-192.pdf)
[\[10\] Zeng, D., Liu, K., Lai, S., Zhou, G., & Zhao, J. (2014). Relation Classification via Convolutional Deep Neural Network. Coling, (2011), 2335–2344.](http://www.aclweb.org/anthology/C14-1220) 
[\[11\] Gao, J., Pantel, P., Gamon, M., He, X., & Deng, L. (2014). Modeling Interestingness with Deep Neural Networks.](http://research.microsoft.com/pubs/226584/604_Paper.pdf)
[\[12\] Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014). A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management – CIKM ’14, 101–110.](http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf) 
[\[13\] Weston, J., & Adams, K. (2014). # T AG S PACE : Semantic Embeddings from Hashtags, 1822–1827.](http://emnlp2014.org/papers/pdf/EMNLP2014194.pdf)
[\[14\] Santos, C., & Zadrozny, B. (2014). Learning Character-level Representations for Part-of-Speech Tagging. Proceedings of the 31st International Conference on Machine Learning, ICML-14(2011), 1818–1826.](http://jmlr.org/proceedings/papers/v32/santos14.pdf) 
[\[15\] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification, 1–9.](http://arxiv.org/abs/1509.01626)
[\[16\] Zhang, X., & LeCun, Y. (2015). Text Understanding from Scratch. arXiv E-Prints, 3, 011102.](http://arxiv.org/abs/1502.01710)
[\[17\] Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2015). Character-Aware Neural Language Models.](http://arxiv.org/abs/1508.06615)





