
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

Markov Chains
