
[원문](http://deeplearning4j.org/understandingRBMs.html)

Restricted Boltzmann machines은 뉴럴넷 르네상스뿐만 아니라, 딥러닝 넷의 심장이다. 그래서 RBM의 메카니즘은 주목할 필요가 있다.

RBM이 동작하는 과정을 공략하기 위해, MNIST의 예제를 사용할 것이다. MNIST는 0에서 9까지 손으로 쓴 숫자 이미지 모음이고, RBMs는 그 이미지를 인식하고 분류할 것이다

각 RBM은 두 개의 "노드" 층을 가진다. 두 층은 가시<sup>visible</sup>층과 은닉<sup>hidden</sup>층이다. 첫번째, 가시층은 학습을 위해 네트워크에 주입되는 입력을 받는다. RBM의 입력을 데이터의 랜덤 샘플이 담기는 그릇으로 생각할 수 있다. They’re boxes, each holding a data point, which in the initial layer would be a sampling of pixels.

To the RBM, each image is nothing more than a collection of pixels that it must classify. 
RBM에서는, 각 이미지는 픽셀의 모음 이상이 아니다 / 분류할
The only way to classify images is by their features, the distinctive characteristics of the pixels. 
이미지를 분류하는 유일한 방법은 / 그들의 피쳐에 의한것이다 / 픽셀의 특유의 성질인
Those are generally dark hues and light ones lined up along edges, curves, corners, peaks and intersections — the stuff that handwritten numerals and their backgrounds are made of, their constituent parts.
그것들은 일반적으로 어두운 색조와 빛이다 / 하나는 가장가지, 곡선, 구석, 

As it iterates over MNIST, an RBM is fed one numeral-image at a time without knowing what it is. 
그것은 MNIST를 반복적으로 읽을때 / RBM는 한번에 하나의 숫자 이미지를 읽는다 / 그것이 무엇인지 아는 것 없이 /
In a sense, the RBM is behind a veil of ignorance, and its entire purpose is to learn what numeral it is dealing with behind the veil by randomly sampling pixels from the numeral’s unseen image, and testing which ones lead it to correctly identify the number. 
어느정도,  RBM은 무지의 장막 건너편에 있다 / 그리고 그것의 전체 목적은 / 어떤 숫자인지 배우는 것이다 / 장막 뒤를 다룸으로써 / 픽셀을 랜덤하게 샘플링해서 / 숫자의 보지 않은 이미지로부터 / 그리고 어떤 것이 숫자의 올바른 인식으로 이끄는지 테스트한다
(There is a benchmark dataset, the test set, that knows the answers, and against which the RBM-in-training contrasts its own, provisional conclusions.)
벤치마크 데이터셋이 있다 / 정답을 아는 테스트셋 / 그리고 훈련의 RBM이 그것 자신의 시험적인 결론을 대조해보는 

Each time the RBM guesses wrong, it is told to go back and try again, until it discovers the pixels that best indicate the numeral they’re a part of – the signals that improve its capacity to classify. 
RBM이 잘못 추론할 때마다 / 되돌아 가서 다시시도하라고 한다 / 그것이 픽셀을 발견할 때까지 / 가장 숫자를 잘 가르키는 / 그들은 신호의 부분이다 / 분류의 능력을 향상시키는 
The connections among nodes that led it to the wrong conclusion are punished, or discounted. 
노드 사이의 연결은 / 잘못된 결론을 이끄는 / 벌칙을 받는다 / 또는 디스카운트된다 /
They grow weaker as the net searches for a path toward greater accuracy and less error.
그들은 약해진다 / 넷이 경로를 찾음에 따라 / 큰 정확도와 적은 에러를 가지는

### Invisible Cities
Since it’s a stretch to even imagine an entity that cannot identify numbers, one way to explain how RBMs work is through analogy.
개체를 상상하도록 과장되었기 때문에 / 숫자를 구별할수 없는 / RBM이 어떻게 동작하는지 설명하는 한가지 방법은 / 비유법을 통하는 것이다.

Imagine each numeral-image like an invisible city, one of ten whose names you know: San Francisco, New York, New Orleans, Seattle… An RBM starts by reaching into the invisible city, touching various points on its streets and crossings. 
각 숫자-이미지를 상상하라 / 보이지 않는 도시 같은 / 10개 중 하나의 이름은 네가 안다: 샌프란시스코, 뉴욕, 뉴 오클랜드, 시애틀 / RBM은 보이지 않은 도시로 도달함으로 시작한다 / 다양한 점을 건드리며 / 거리와 교차로 위의
If it brought back a “Washington St.,” one of the most common street names in America, it would know little about the city it sought to identify. 
“Washington St.,”로 돌아오면/ 미국에서 가장 흔한 거리 이름의 하나인 / 그것은 도시에 대해서 거의 알지 못한다 /  인지하도록 찾는 
This would be akin to sampling an image and bringing back pixels from its black background, rather than an edge that formed one of the numeral’s lines or kinks. 
이는 이미지 샘플링과 비슷하다 / 그리고 픽셀을 다시 가져오는 것과 비슷하다 / 검은 배경으로 부터 / 가장자리보다 / 숫자의 선과 비틀림의 하나를 형성하는 
The backdrop tells you almost nothing useful of the data structure you must classify. 
배경은 너에게 데이터 구조의 유용한 어떤것도 말해주지 못한다 / 네가 분류해야할 
Those indistinguishable pixels are Washington St.
그러한 구별불가능한 픽셀은 Washington St이다.

Let’s take Market St instead… New York, New Orleans, Seattle and San Francisco all have Market Streets of different lengths and orientations.
"Market St"를 대신 취하자 / 뉴욕, 뉴 오클랜드, 시애틀, 샌프란시스코 모두  Market Streets를 갖는다 / 다른 길이와 방향의 
 An invisible city that returns many samples labeled Market Street is more likely to be San Francisco or Seattle, which have longer Market Streets, than New York or New Orleans.
보이지 않는 도시 / 많은 샘플을 리턴하는 / Market Street로 레이블링된 / 샌프란시스코 또는 시애틀일확률이 크다 / 긴 Market Streets를 가진 / 뉴욕과 뉴 오클랜드보다

By analogy, 2s, 3s, 5s, 6s, 8s and 9s all have curves, but an unknown numeral-image with a curve as a feature is less likely to be a 1, 4 or 7. 
비유에 의해서, 2s, 3s, 5s, 6s, 8s and 9s는 모두 곡선을 갖는다 / 그러나 알려지지 않은 숫자-이미지은 / 곡선을 가진 / 1, 4, 7일 가능성이 적다. 
So an RBM, with time, will learn that curves are decent indicators of some numbers and not others. 
그래서, RBM은 / 시간을 가진 / 학습할 것이다 / 곡선은 좋은 척도이다 / 어떤 숫자의 / 그리고 다른 것은 아닌 
They will learn to weight the path connecting the curve node to a 2 or 3 label more heavily than the path extending toward a 1 or 4.
그들은 / 학습할 것이다 / 경로에 가중치를 준다 / 곡선노드를 연결하는 / 2 또는 3 레이블에 / 좀 더 높은 / 경로보다 / 1 과 4를 확장하는 

In fact, the process is slightly more complex, because RBMs can be stacked to gradually aggregate groups of features, layer upon layer. 
사실, 과정은 / 좀 더 복잡하다 / 왜냐하면 / RBMs은 쌓일수 있다 / 피쳐의 그룹을 점진적으로 합치기 위해 / 층층이
These stacks are called deep-belief nets, and deep-belief nets are valuable because each RBM within them deals with more and more complex ensembles of features until they group enough together to recognize the whole:
이러한 스택은 딥-빌리프 넷이라고 불린다 / 그리고 딥-빌리프 넷은 유용하다 / 왜냐하면 그들 안에 있는 각 RBM은 좀 더 복잡한 피쳐의 앙상블을 다루기 때문이다 / 그들이 충분히 그룹핑할 때까지 / 전체를 인지하기 위해 
 pixel (input), line, chin, jaw, lower face, visage, name of person (label).
:픽셀, 선, 턱, 얼굴, 이름 
![enter image description here](http://deeplearning4j.org/img/feature_hierarchy.png)

But let’s stick with the city and take just two features together. 
도시로 붙자 / 그리고 두 개의 피쳐를 함께 취하자.
If an urban data sample shows an intersection of Market and an avenue named Van Ness, then the likelihood that it is San Francisco is high. 
도시 데이터 샘플은 Market과 Van Ness이름의 길의 교차로를 보여준다면, 확률은 / 그것이 샌프란시스코일 / 매우 높다.
Likewise, if data samples from the numeral-image show a vertical bar meeting a partial circle that opens to the left, then we very likely have a 5 and nothing else.
비슷하게, 데이터 샘플이 / 숫자-이미지로부터 / 수직 막대기를 보여준다면 / 부분 동그라미를 만나는 / 왼쪽이 열린 / 5일 확률이 크다. 

Now let’s imagine both the numeral-images and invisible cities as maps whose points are connected to each other as probabilities. 
상상해보자 / 숫자-이미지와 보이지 않는 도시 둘다 / 서로 연결됐다고 / 확률로써
If you start from a curve on an 8 (even if you don’t know it’s an 8), the probability of landing on another curve at some point is nearly 100%;
8에서의 곡선으로 시작하면(네가 심지어 8을 알지 못해도), 어떤 점에서 다른 곡선에 도착할 확률은 거의 100프로이다.
 if you are on a five, that probability is lower.
 5에 있다면, 확률은 낮을 것이다.

Likewise, if you start from Market in San Francisco, even if you don’t know you are in San Francisco, you have a high probability of hitting Van Ness at some point, given that the two streets bisect the city and cross each other at the center.
비슷하게, San Francisco의 Market으로 시작하면, 네가 San Francisco에 있는지 모르더라도 / Van Ness에 도착할 확률이 크다 / 어떤 점에서 / 주어지면 / 두 개의 거리가 도시를 이등분하고 / 서로 교차된다 / 어떤점에서 

### Markov Chains

RBMs tie all their nodes together in an algorithm called a Markov Chain. 
RBM은 모든 노드를 묶는다 / 마르코프 체인이라는 알고리즘으로
Markov Chains are essentially logical circuits that connect two or more states via probabilities. 
마르코프 체인은 논리적인 회로이다 / 두 개 또는 그 이상의 상태를 연결하는 / 확률로써
A sequence of coin flips, a series of die rolls, [Rozencrantz and Guildenstern marching toward their fate](https://en.wikipedia.org/wiki/Rosencrantz_and_Guildenstern_Are_Dead).
동전 던지기의 순서 / 주사위 던지기의 연속 / 

Let’s explore this idea with another absurdly long analogy.
이 생각을 탐험하자 / 다른 터무니 없이 긴 비유로

We’ll imagine a universe where you have three possible locations, or states, which we’ll call home, office and bowling alley. 
우주를 상상할 것이다 / 세가지 가능한 위치가 있는 / 또는 상태 / 집, 사무실 그리고 볼링장이라 부를 것이다
Those three states are linked by probabilities, which represent the likelihood that you’ll move from one to the other.
그러한 세가지 상태는 연결된다 / 확률에 의해 / 가능성을 표현하는 / 너가 움직일 / 하나에서 다른 하나로 

At any given moment when you’re home, there is a low probability of you going to the bowling alley, let’s say 10%, a midsize one of going to the office, 40%, and a high one of you remaining where you are, let’s say 50%. 
어떠한 주어진 상태에서 / 당신이 집에 있을 때 / 작은 확률이 있다 / 당신이 볼링장을 가는 / 그것을 10%라고 하자 / 사무실로 가는 확률 40% / 그리고 나머지는 그냥 집에 있는 확률 50%
The probabilities exiting any one state should always add up to 100%.
확률은 / 어떤 상태로 가는 / 항상 합해서 100%이여야 한다.

Now let’s take the bowling alley: At any given moment while you’re there, there’s a low probability of you remaining amid the beer and slippery shoes, since people don’t bowl for more than a few hours, a low one of going to the office, since work tends to come before bowling, and a high one of you returning home. 
볼링장을 취해보자 / 상태가 주어졌을 때 / 당신이 거기 있는 동안 / 작은 확률이 있다 / 당신이 맥주와 슬리퍼를 신고있을 / 사람들은 볼링을 하지 않기 때문이다 / 몇 시간 이상 / 사무실로 가는 것도 낮다 / 왜냐하면 일은 볼링 전에 오는 경향이 있다 / 집에 갈 확률은 높다 / 
Let’s make those 20%, 10% and 70%.
그 것들을 20%, 10%  70%라 하자.

So a home state is a fair indication of office state, and a poor indication of bowling alley state. 
집 상태는 사무실 상태의 좋은 지표이다 / 그리고 볼링장의 나쁜 지표이다.
While bowling alley state is a great indicator of home state and a poor indicator of office state. 
볼링장 상태는 집 상태의 좋은 지시자이고 / 사무실 상태의 나쁜 지시자이다
(We’re skipping office state because you get the point.) 
사무실 상태는 스킵하고 있다 / 왜냐하면 당신이 요점을 이해했기 때문에
Each state is in a garden of forking paths, but the paths are not equal.
각 상태는 / 갈라지는 경로의 정원이다 / 그러나 그 경로는 동일하지 않다.

Markov Chains are sequential. 
마르코프 체인은 연속적이다.
Their purpose is to give you a good idea, given one state, of what the next one will be. 
그들의 목적은 / 당신에게 좋은 아이디어를 주는 것이다 / 상태가 주어지면 / 다음은 무슨일이 있어나는지
Instead of home, office and bowling alley, those states might be edge, intersection and numeral-image, or street, neighborhood and city. 
집 대신에, 사무실과 볼링장, 그러한 상태는 / 가장자리, 교차, 숫자-이미지, 이웃, 도시 일수 있다.
Markov Chains are also good for predicting which word is most likely to follow a given wordset (useful in natural-language processing), or which share price is likely to follow a given sequence of share prices (useful in making lots of money).
마르코프 체인은 또한 예측을 잘한다 / 어떤 단어가 올지 / 다음에 / 단어셋이 주어지면 / (자연어 처리에 유용하다) / 또한 어떠한 공유 가격이 올지 / 공유  가격의 순서가 주어졌을 때 / (많은 돈을 버는데 유용하다)

Remember that RBMs are being tested for accuracy against a benchmark dataset, and they record the features that lead them to the correct conclusion.
RBM은 정확률이 테스트 됨을 기억하라 / 벤치마크 데이터 셋에 대해서 / 그리고 / 그들은 피쳐를 기록한다 / 그들을 이끄는 / 올바른 결론을 이끄는 
 Their job is to learn and adjust the probabilities between the feature-nodes in such a way that if the RBM receives a certain feature, which is a strong indicator of a 5, then the probabilities between nodes lead it to conclude it’s in the presence of a 5.
그들의 직업은 학습하는 것이고 / 확률을 조절한다 / 피쳐-노드 사이의 / 그러한 방법으로 / RBM이 특정 피쳐를 받는다면 / 5의 강한 지시자인 / 그러면 노드 사이의 확률은 / 결론을 이끈다 / 그것이 5이라는 
 They register which features, feature groups and numeral-images tend to light up together.
 그들은 등록한다 / 어떤 피쳐, 피쳐 그룹 그리고 숫자-이미지가 같이 환해지는지 

Now, if you’re ready, we’ll show you how to implement a [deep-belief network](http://deeplearning4j.org/deepbeliefnetwork.html).