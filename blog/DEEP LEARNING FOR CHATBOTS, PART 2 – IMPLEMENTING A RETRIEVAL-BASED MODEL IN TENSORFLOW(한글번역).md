## DEEP LEARNING FOR CHATBOTS, PART 2 – IMPLEMENTING A RETRIEVAL-BASED MODEL IN TENSORFLOW(한글번역)
@[published]

다음 [포스트](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)를, 좀 더 자세히 읽으려는 공부목적으로 번역해보았습니다. 

[The Code and data for this tutorial is on Github.](https://github.com/dennybritz/chatbot-retrieval/)

##### 검색기반 봇 RETRIEVAL-BASED BOTS
이 포스트에서 검색기반 봇을 구현할 것이다. 검색 기반 모델은 이전에 보지 못했던 응답을 생성하는 생성모델과 달리 선-정의<sup>pre-define</sup>된 응답의 저장소이다.  좀 더 형식화하면, 검색 기반 모델의 입력은 문맥 $c$(이 시점까지 대화) 와 잠재적 응답 $r$  이다. 모델 출력은 응답에 대한 점수이다. 좋은 응답을 찾기 위해선, 여러 개의 응답에 대한 점수를 계산하고, 가장 높은 점수를 골라야 한다. 그러나, 생성모델을 만들 수 있다면, 왜 검색기반 모델을 만들어야 하는가? 선정의된 응답의 저장소가 필요없기 때문에, 생성 모델이 좀더 유연해보인다. 

문제는 생성모델은 최소한 아직까지는 실제에서는 잘 동작하지 않는다. 왜냐하면, 생성모델은 응답에 대한 자유가 커서, 문법적 실수를 하거나, 연관없고, 일반적이고, 일치하지 않는 응답을 하는 경향이 있다. 생성모델은 수많은 훈련 데이터가 필요하고, 최적화하기도 어렵다. 제품 시스템의 대부분은 검색-기반이거나 검색-기반과 생성모델의 결합이다.

구글의 [Smart Reply](http://arxiv.org/abs/1606.04870)가  좋은 예이다. 생성모델은 활발한 연구분야이지만, 아직 실용화단계까지는 가지 못했다. 당신이 대화형 에이전트를 만들기를 원한다면, 가장 좋은 방식은 검색-기반 모델일 확률이 높다.

우분투 대화 코퍼스 THE UBUNTU DIALOG CORPUS
이 포스트에서는 Ubuntu Dialog Corpus로 일할 것이다. ([paper](http://arxiv.org/abs/1506.08909), [github](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)). 

Ubuntu Dialog Corpus (UDC) 는 이용가능한 가장 큰 공개 대화 데이터셋 중 하나이다. 공개 IRC 네트워크의 우분투 채널에서의 대화 로그를 기반으로 한다. 논문은 corpus가 어떻게 생성됐는지에 대해서 다룬다. 그래서 여기서는 다루지 않는다. 그러나, 어떤데이터를 다루게 되는지 이해하는 것은 중요하다. 약간의 탐험을 해보자. 훈련 데이터는 1,000,000 예제와 50% 긍정 (label 1) and 50% 부정 (label 0)으로 이루어져있다. 각 예제는 문맥과, 그 시점까지의 대화, 발언<sup>utterance</sup>, 문맥에 대한 응답으로 이루어져 있다.

 긍정은 발언이 실제 문맥에 대한 실제 응답이라는 의미이고, 부정은 틀렸다는 뜻이다. 부정은 코퍼스 내 어딘가에서 랜덤하게 뽑는다. 여기에 샘플 데이터가 있다.

![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/Screen-Shot-2016-04-20-at-12.29.42-PM.png)

데이터셋 생성 스크립트는 이미 많은 전처리를 해놓았다 - [NLTK](http://www.nltk.org/)를 이용한  [토크나이즈](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize), [스테밍](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball), [렘마타이즈](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet) -  스크립트는 이름, 위치. 기관, URL, 시스템 경로같은 엔티티를 특수 token으로 변환한다. 이 전처리는 꼭 필요하지는 않지만, 어느정도 성능 향샹을 할 수 있게 해준다. 평균 문맥 길이는 86단어이고 평균 발언 길이는 17 단어이다. [데이터 분석을 보기 위해선 쥬피터 노트북을 확인하라.](https://github.com/dennybritz/chatbot-retrieval/blob/master/notebooks/Data%20Exploration.ipynb)

데이터에는 테스트와 검증셋이 딸려있다. 이것들의 포맷은 훈련 데이터의 포맷과는 다르다. 각 test/validation set의 각 레코드는 문맥, 정답 발언과 distractors로 불리는 9개의 틀린 발언으로 이루어져 있다. 모델의 목적은 정답 발언에 가장 높은 점수를 주고 틀린 발언에는 낮은 점수를 주는 것이다.

![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/Screen-Shot-2016-04-20-at-12.43.09-PM.png)

모델이 얼마나 좋은지 평가하는 여러가지 방법이 있다. 주로 사용하는 방법은 recall@k이다. Recall@k는 모댈아 10개의 가능한 응답 중에 k개의 가장 좋은 응답을 고르게 한다. 정답 응답하나가 고른 것들 중에 있다면, 그 테스트 예제는 맞다고 표시한다. 그래서, k가 커진다는 것은 이 task가 쉬워진다는 것이다. 데이터셋에는 10개의 응답밖에 없기 때문에, k=10라면  100%의 recall을 얻는다. k=1이면, 모델은 정답 응답을 고를 단 한번의 기회밖에 없다.

이 시점에서 어떻게 9 distractors를 골랐는지 궁금할 것이다. 이 데이터셋에서 9 distractors는 랜덤하게 골라졌다. 하지만, 실제 세계에서는 몇 백만개의 가능한 응답이 있을 수 있고, 어느것이 옳은 것인지 모른다. 가장 높은 점수의 응답을 고르기 위해 백만의 가능한 응답을 평가할 수 없다. 그 일은 너무 비싸다.

Google’s Smart Reply uses clustering techniques to come up with a set of possible responses to choose from first.  구글의 Smart Reply는 가능한 응답의 집합을 얻기 위해 클러스터링 기법을 사용한다. 몇 백개의 가능한 응답이 있다면, 그것을 모두 평가할 수 있을 것이다.

##### BASELINES
뉴럴 네트워크 모델을 시작하기 전에, 어떤 종류의 성능을 원하는지 이해하기 위해 간단한 baseline 모델을 만들자. `recall@k` metric를 평가하기 위해 다음 함수를 사용할 것이다.

```
def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples
```
`y`는 내림차순으로 정렬된 예측의 리스트이다. `y_test`는 실제 레이블이다. 예를 들어,  `[0,3,1,2,5,6,4,7,8,9]` 의 `y` 는 발언 0이 가장 높은 점수를 얻었고, 발언 9가 낮은 점수를 얻었다는 의미이다. 각 테스트 예제에 대해서 10개의 발언를 가졌고, 첫번째 (index 0)은 항상 정답이다. 왜냐하면 발언 column은 distractor columns 앞에 오기 때문이다. 직관적으로, 완전 랜덤 예측기는 `recall@1`에 대해서 10%를 얻고, `recall@2`는 20%를 얻는다. 이 것이 사실인지 보자.

```
# Random Predictor
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)
# Evaluate Random predictor
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
y_test = np.zeros(len(y_random))
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))
```

```
Recall @ (1, 10): 0.0937632
Recall @ (2, 10): 0.194503
Recall @ (5, 10): 0.49297
Recall @ (10, 10): 1
```

좋다. 잘 동작하는 것처럼 보인다. 물론, 우리가 랜덤 예측기를 원하는 것은 아니다. 원 논문에서 논의한 다른 baseline은 tf-idf 예측기이다. [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 는 “term frequency – inverse document frequency”를 의미하고, 문서에서의 단어가 전체 문서집합에서 상대적으로 얼마나 중요한지를 측정한다. 자세한 사항은 논외로 하고, 비슷한 내용을 가진 문서는 비슷한 tf-idf 벡터를 가질 것이다. 

직관적으로, 문맥과 응답이 비슷한 단어를 가지고 있다면, 그 둘은 올바른 쌍일 가능성이 크다. 최소한 랜덤보다는 가능성이 높다. ([scikit-learn](http://scikit-learn.org/) 같은) 많은 라이브러리는 내장 tf-idf 함수가 딸려있다. 매우 쓰기 쉽다. tf-idf 예측기를 만들고 얼마나 잘 동작하는지 보자.

```
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
 
    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))
 
    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]
```

```
# Evaluate TFIDF predictor
pred = TFIDFPredictor()
pred.train(train_df)
y = [pred.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))
```

```
Recall @ (1, 10): 0.495032
Recall @ (2, 10): 0.596882
Recall @ (5, 10): 0.766121
Recall @ (10, 10): 1
```

tf-idf 모델은 랜덤모델보다 매우 낫다. 하지만 완벽하지는 않다. tfidf 모델의 가장은 그렇게 좋지는 않다. 무엇보다도, 응답이 정답이기 위해서 꼭 문맥과 비슷할 필요는 없다. 둘째로, tf-idf는 중요한 신호가 될 수 있는 단어 순서를 무시한다. 뉴럴네트워크 모델과 함께라면, 좀 더 그 일을 잘할 수 있다.

##### DUAL ENCODER LSTM
이 포스트에서 우리가 만들 딥러닝 모델은 Dual Encoder LSTM network라 불린다. 이 타입의 네트워크는 이 문제에 적용할 수 있는 많은 모델 중 하나이며, 반드시 가장 좋은 것은 아니다. 당신은 아직 시도해보지 못한 모든 종류의 딥러닝 구조를 생각해 볼 수 있다. - 그것은 매우 활발한 연구 분야이다. 예를 들어, 기계 번역 분야에서 자주 쓰이는 [seq2seq](https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html)은 이 문제에 아마도 잘 맞을 것이다.

Dual Encoder를 다루는 이유는 이 데이터셋에 대해서 매우 잘 동작한다고 [보고](http://arxiv.org/abs/1510.03753)  되었기 때문이다. 이것은  우리가 기대하는 것이고 우리의 구현이 틀리지 않았음을 확신할수도 있다. 이 문제에 대한 다른 모델의 적용은 흥미로운 프로젝트이다. 우리가 만들 Dual Encoder LSTM는 이와 같다.
![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/Screen-Shot-2016-04-21-at-10.51.18-AM.png)

이 모델은 대충 이렇게 동작한다.

1. 문맥과 응답 텍스트 모두 단어로 나뉜다. 각 단어는 벡터로 임베딩된다. 워드 임베딩은 스탠포드의 GloVe vectors로 초기화 되고 훈련기간동안 fine-tuned된다 (이 과정은 선택사항이며, 그림에는 보여져 있지 않다. 필자는 Glove의 단어 임베딩을 사용하는 것은 모델 성능에는 별로 영향을 끼치지 않는다는 것을 발견했다)

2. 임베딩된 문백과 응답 모두 같은 Recurrent Neural Network에 단어별로 주입된다. RNN은 벡터 표현을 생성한다. 대충 말하자면, 문맥과 응답의 의미를 잡는다(그림에서 $c$와 $r'$). 얼마나 큰 벡터를 사용할 것인지는 고를 수 있다. 여기서는 256 차원이다.

3. 응답 $r$에 대한 "예측"을 하기 위해 행렬 $M$에 $c$를 곱한다. $c$가 256차원 벡터이면,  $M$은 256×256 차원 행렬이고, 결과는 다른 256차원 벡터이다. 그 결과 벡터는 생성된 응답으로 해석할 수 있다. $M$은 훈련시간에 학습된다.

4. 두 벡터의 내적을 취함으로써, 예측된 응답 $r'$와 실제 응답 $r$ 사이의 유사도를 측정한다. 큰 내적은 벡터가 유사하고 응답은 높은 점수를 얻는다. 스코어를 확률로 바꾸기 위해, sigmoid함수를 적용한다. 3, 4 단계는 그림에서 서로 결합되어 있음을 주시하라.
 
네트워크를 훈련하기 위해, 손실(비용) 함수가 필요하다. 분류문제에 흔히 쓰이는 binary cross-entropy loss를 사용할 것이다.
문맥-응답 쌍에 대한 정답 레이블을  `y`라 부르자. 1 (actual response) 또는  0 (incorrect response)이 될 수 있다. 4에서 예측 확률을 `y'`라 하자.  그 후, cross entropy loss는 `L= −y * ln(y') − (1 − y) * ln(1−y)`로 계산된다. 이 식 뒤에 있는 직관은 쉽다. `y=1`이면 `L = -ln(y')`만 남고, 1에서 멀어진 예측에 대한 페널티를 받는다. `y=0`이면 `L= −ln(1−y)`만 남고 0에서 멀어진 것에 대한 페널티를 받는다. 여기서 구현에서는 numpy, pandas, Tensorflow, TF Learn(텐서플로우의 고레벨의 수준의 라이브러리)의 조합을 사용한다.

##### DATA PREPROCESSING
데이터셋은 원래 CSV 포맷으로 되어 있다. 

We could work directly with CSVs, but it’s better to convert our data into Tensorflow’s proprietary Example format. (Quick side note: There’s also tf.SequenceExample but it doesn’t seem to be supported by tf.learn yet). 
CSV로 작업할수도 있지만, 텐서플로우의 독점 [예제](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) 포맷을 사용하는것이 더 낫다(Quick side note: There’s also `tf.SequenceExample` but it doesn’t seem to be supported by tf.learn yet). 입력 파일에서 바로 텐서를 읽을 수 있고 텐서플로우가, 입력의셔플링, 배칭, 큐잉을 할수 있게 해준다. 또한, 전처리의 부분으로써, vocabulary를 생성한다. 이는 각 단어를 정수 숫자에 매핑하는 것을 의미한다. 예를 들어 “cat”은 `2631`이 될 수 있다. 생성할 `TFRecord` 파일은 단어 스트링 대신 이 정수 숫자를 저장한다. 또한 정수에서, 단어로 다시 매핑할 수 있는 단어집도 저장한다.

각 `Example`는 다음 필드를 포함한다.

`context`: 문맥 텍스트를 표현하는 단어 id의 순서 e.g. `[231, 2190, 737, 0, 912]`
`context_len`: The length of the context, e.g. 5 for the above example
`context_len`: 문맥의 길이  e.g.  위의 예에서 `5`
`utterance` 발언 텍스트를 표현하는 단어 id의 순서 (response)
`utterance_len`: 발언의 길이
`label`:  훈련셋에만 있다.  0 or 1.
`distractor_[N]`: the test/validation data에만 있다.  `N` 은 0~8까지 있다.  발언을 표현하는 단어 ids의 순서
`distractor_[N]_len`: distractor utterance의 길이

전처리는 [prepare_data.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/scripts/prepare_data.py) 에서 처리된다. 3개의 파일 `train.tfrecords`, `validation.tfrecords` , `test.tfrecords`이 만들어진다. 당신이 직집 이 스크립트를 [다운로드](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM)하여 수행해볼수 있다.

##### CREATING AN INPUT FUNCTION
훈련과 평가에 대한 텐서플로우의 내장 지원을 사용하기 위해, 텐서플로우의 (입력 데이터의 배치를 리턴하는) 입력 함수를 생성해야한다.  사실, 훈련과 테스트는 다른 포맷이기 때문에, 각각에 대한 다른 입력함수가 필요하다. 입력 함수는 피쳐와 레이블(가능하면)의 배치를 리턴해야한다. Something along the lines of:
```
def input_fn():
  # TODO Load and preprocess data here
  return batched_features, labels
```

훈련과 평가동안 다른 입력함수가 필요하기 때문에 그리고 코드 중복을 증오하기 때문에, 적절한 모드에 대한 입력함수를 생성하는 `create_input_fn`라 불리는 wrapper를 만든다. It also takes a few other parameters. Here’s the definition we’re using:
```
def create_input_fn(mode, input_files, batch_size, num_epochs=None):
  def input_fn():
    # TODO Load and preprocess data here
    return batched_features, labels
  return input_fn
```
전체 코드는 [udc_inputs.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_inputs.py)에 있다. 고레벨에서는 함수는 다음과 같은 일을 한다.

1. `Example` 파일의 필드를 설명하는 feature 정의를 만든다.
1. `tf.TFRecordReader`로  `input_files`로부터 레코드를 읽는다
1. feature 정의에 따라 레코드를 파싱한다.
1. 훈련 레이블을 추출한다.
1. 다수의 예제와 훈련 레이블을 배치로 나눈다.
1. 배치로 나뉜 예제와 훈련레이블을 리턴한다.

##### DEFINING EVALUATION METRICS
모델을 평가하기 위해 `recall@k` metric를 사용하기를 원한다는 것을 이미 언급했다. 운이 좋게도, 텐서플로우는 `recall@k`를 포함한 표준 평가 메트릭들이 딸려있다. 이 메트릭을 사용하기 위해, 메트릭 이름을 예측치와 정답 레이블을 인자로 받는 함수로 매핑하는 dictionary를 생성한다.
```
def create_evaluation_metrics():
  eval_metrics = {}
  for k in [1, 2, 5, 10]:
    eval_metrics["recall_at_%d" % k] = functools.partial(
        tf.contrib.metrics.streaming_sparse_recall_at_k,
        k=k)
  return eval_metrics
```
위 코드에서 3개의 인자를 받는 함수를 2개의 인자를 받는 함수로 변환하는 [functools.partial](https://docs.python.org/2/library/functools.html#functools.partial)를 사용한다. `streaming_sparse_recall_at_k`이름에 헷갈려하지 마라. Streaming은 단지, 여러 개의 배치에 대해서 쌓인다는 의미이고 sparse는 레이블의 포맷을 말한다.

이는 중요한 점을 시사한다. 평가 시간동안, 예측의 정확한 포맷은 무엇인가? 학습시간 동안, 예제가 정답일 확률을 예측한다. 그러나, 평가 시간에는, 목표는 발언과 9개의 distractors에 대해서 점수를 매기고, 가장 좋은 것을 고른다 - 단지 정답/틀림을 예측하는것이 아닌다. 이는 평가 시간에, 각 예제는 `[0.34, 0.11, 0.22, 0.45, 0.01, 0.02, 0.03, 0.08, 0.33, 0.11]`같은 10개 점수의 벡터를 출력해야 한다는 것이다. 각 점수는 정답 응답과 9개의 distractors에 대응한다. 각 발언은 독립적으로 점수가 매겨지므로, 확률은 합이 1일 필요는 없다. 정답 응답은 항상 배열에서 0번째 있기 때문에, 각 예제에 대한 레이블은 0이다. 정답 응답에 대해서는 0.34를 받은 반면, 세번째 distractor가 0.45의 확률을 얻었으므로,  `recall@1`에 의해서는 이 예제는 틀렸다. `recall@2`에 의해서는 이 예제는 정답이다.

### 학습 코드를 표준화하기(BOILERPLATE TRAINING CODE )
뉴럴 네트워크 코드를 작성하기 전에, 모델을 학습하고 평가하는 것을 표준화한 코드를 작성하겠다. 이는 올바른 인터페이스를 유지하는 한, 당신이 사용할 네트워크를 교체하기 쉽게 해준다. 입력으로 배치화된 피쳐, 레이블, 모드(훈련 또는 평가)를 받고 예측을 출력하는 모델 함수 `model_fn`를 가정하자.  그 후, 다음과 같이 모델을 훈련하는 일반적 목적의 코드를 작성할 수 있다.

```python
estimator = tf.contrib.learn.Estimator(
model_fn=model_fn,
model_dir=MODEL_DIR,
config=tf.contrib.learn.RunConfig())
 
input_fn_train = udc_inputs.create_input_fn(
mode=tf.contrib.learn.ModeKeys.TRAIN,
input_files=[TRAIN_FILE],
batch_size=hparams.batch_size)
 
input_fn_eval = udc_inputs.create_input_fn(
mode=tf.contrib.learn.ModeKeys.EVAL,
input_files=[VALIDATION_FILE],
batch_size=hparams.eval_batch_size,
num_epochs=1)
 
eval_metrics = udc_metrics.create_evaluation_metrics()
 
# We need to subclass theis manually for now. The next TF version will
# have support ValidationMonitors with metrics built-in.
# It's already on the master branch.
class EvaluationMonitor(tf.contrib.learn.monitors.EveryN):
def every_n_step_end(self, step, outputs):
  self._estimator.evaluate(
    input_fn=input_fn_eval,
    metrics=eval_metrics,
    steps=None)
 
eval_monitor = EvaluationMonitor(every_n_steps=FLAGS.eval_every)
estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])
```

여기서 `model_fn`에 대한 estimator를 생성한다. 훈련과 평가에 대한 두 입력 함수, 평가 메트릭 사전을 입력으로 받는다. 훈련기간동안 `FLAGS.eval_every` 단계마다 모델을 평가하는 monitor를 정의한다. 마침내, 모델을 학습한다. 학습은 막연하게 수행되지만, 텐서플로우는 `MODEL_DIR`에 체크포인트 파일을 자동적으로 저장한다. 그래서 당신은 훈련을 언제든 중단할 수 있다. 좀 더 fancy한 테크닉은 검수셋 메트릭이 더 나아지지 않을 때(오버피팅이 시작될 때)자동적으로 학습을 중단하는 early stopping를 쓰는 것이다. 전체 코드는  [udc_train.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_train.py).

간단하게 언급하고자 하는 것은 `FLAGS`의 사용법이다. 이는 파이썬 `argparse`와 비슷하게 프로그래에 커맨드라인 파라메터을 주는 방법이다. `hparams`는 모델을 튜닝할 수 있는 하이퍼파라메터를 저장하는 [hparams.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_hparams.py) 에 만든 커스텀 객체이다. 이 `hparams` object 는 모델을 인스턴스화할 때 주어진다.

##### CREATING THE MODEL
입력, 파싱, 평가, 훈련에 대한 표준화한 코드를 만들었으므로,  Dual LSTM neural network에 대한 코드를 작성할 시간이다.훈련과 평가에 대해서 서로 다른 포맷을 가지고 있기 때문에, 데이터를 올바른 포맷으로 가져오는 일을 하는  [`create_model_fn`](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_model.py) 래퍼를 작성했다. 실제 예측을 하는 `model_impl`를 인자로 받는다. 우리의 경우에는,  예측은 위에서 설명한 Dual Encoder LSTM가 한다. 하지만  다른 뉴럴 네트워크로 쉽게 교체할 수있다. 어떻게 생겼는지 보자.

```
def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):
 
  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)
 
  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")
 
 
  # Build the RNN
  with tf.variable_scope("rnn") as vs:
    # We use an LSTM Cell
    cell = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)
 
    # Run the utterance and context through the RNN
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        tf.concat(0, [context_embedded, utterance_embedded]),
        sequence_length=tf.concat(0, [context_len, utterance_len]),
        dtype=tf.float32)
    encoding_context, encoding_utterance = tf.split(0, 2, rnn_states.h)
 
  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
      shape=[hparams.rnn_dim, hparams.rnn_dim],
      initializer=tf.truncated_normal_initializer())
 
    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_utterance = tf.expand_dims(encoding_utterance, 2)
 
    # Dot product between generated response and actual response
    # (c * M) * r
    logits = tf.batch_matmul(generated_response, encoding_utterance, True)
    logits = tf.squeeze(logits, [2])
 
    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)
 
    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))
 
  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
```

전체 코드는 [dual_encoder.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/models/dual_encoder.py)에 있다. 이것이 주어지면, 이전에 정의한 [udc_train.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_train.py)에 있는 main routine의 model function을 인스턴스화할 수 있다.

```
model_fn = udc_model.create_model_fn(
  hparams=hparams,
  model_impl=dual_encoder_model)
```

됐다!.  `udc_train.py`를 실행할 수 있고, 네트워크를 훈련하며, 때때로, 검수 데이터에 대한 recall을 평가한다(`--eval_every`를 바꿈으로써 주기를 조절할 수 있다). `tf.flags`와 `hparams`를 사용하는 정의된 사용가능한 모든 command line flags를 얻기 위해서 `python udc_train.py --help`를 실행해라

```
INFO:tensorflow:training step 20200, loss = 0.36895 (0.330 sec/batch).
INFO:tensorflow:Step 20201: mean_loss:0 = 0.385877
INFO:tensorflow:training step 20300, loss = 0.25251 (0.338 sec/batch).
INFO:tensorflow:Step 20301: mean_loss:0 = 0.405653
...
INFO:tensorflow:Results after 270 steps (0.248 sec/batch): recall_at_1 = 0.507581018519, recall_at_2 = 0.689699074074, recall_at_5 = 0.913020833333, recall_at_10 = 1.0, loss = 0.5383
...
```

##### EVALUATING THE MODEL

모델을 학습한 후, `python udc_test.py --model_dir=$MODEL_DIR_FROM_TRAINING,`를 사용해서 모델을 평가할 수 있다. e.g. `python udc_test.py --model_dir=~/github/chatbot-retrieval/runs/1467389151.` 검수셋 대신에, 테스트셋에 대한 `recall@k` 평가 메트릭을 실행할 수 있을 것이다. 훈련 기간 중에 사용된 같은 파라메터로 `udc_test.py`를 수행해야한다. `--embedding_size=128`로 학습했다면, 같은 것로 테스트 스크립트를 호출해야한다.

After training for about 20,000 steps (around an hour on a fast GPU) our model gets the following results on the test set:
빠른 GPU에서 한시간 정도 걸리는, 대략 20,000 스텝의 훈련 후, 모델은  테스트셋에 대해서 다음 결과를 내놓는다.
```
recall_at_1 = 0.507581018519
recall_at_2 = 0.689699074074
recall_at_5 = 0.913020833333
```
`recall@1`에 대해서는 TFIDF model과 성능이 비슷하지만, `recall@2` 과 `recall@5`는 매우 성능이 좋다. 뉴럴 네트워크가 정답에 높은 점수를 준다는 것이다. 원논문에서는 `recall@1`, `recall@2`, 과 `recall@5`에 대해서 각각 0.55, 0.72 와 0.92를 보고 했다. 하지만 그와 같은 높은 점수는 만들 수 없었다. 아마도 추가적인 전처리와 하이퍼파라메터 최적화가 점수를 올릴수 있을 것이다.

##### MAKING PREDICTIONS

[udc_predict.py](https://github.com/dennybritz/chatbot-retrieval/blob/master/udc_predict.py)를 여태 보지 않았던 데이터에 대해서 예측할 수 있도록 고칠수 있다. 예를 들어 `python udc_predict.py --model_dir=./runs/1467576365/` 출력


```
Context: Example context
Response 1: 0.44806
Response 2: 0.481638
```
You could imagine feeding in 100 potential responses to a context and then picking the one with the highest score.

##### CONCLUSION
이 포스트에서 대화 문맥이 주어졌을때, 잠재 응답에 점수를 매기는 retrieval-based neural network model를 구현했다. 아직까지 발전시킬 여지는 많지만, 다른 뉴럴 네트워크가 dual LSTM encoder보다 더 성능이 좋을 것으로 예측할 수있다. hyperparameter 튜닝, 전처리에 대해서도 발전여지가 있다.  [전체코드는 깃헙에 있고 확인할 수 있다.](https://github.com/dennybritz/chatbot-retrieval/).













