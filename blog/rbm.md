## A Beginner’s Tutorial for Restricted Boltzmann Machines
[원문](http://deeplearning4j.org/restrictedboltzmannmachine.html)

Geoff Hinton에 의해서 만들어진, Restricted Boltzmann machine은 차원축소, 분류, 회귀, 협업 필터링, 피쳐 엔지니어링, 토픽 모델링에 유용하다.

상대적인  심플함과 역사적 중요성 때문에, RBM은 우리가 다룰 첫번째 뉴럴넷이다. 밑에 문단에서, 그림과 쉬운언어로 rbm이 어떻게 동작하는지 설명한다.
![enter image description here](http://deeplearning4j.org/img/two_layer_RBM.png)

위의 그래프에서 동그라미는 노드라 불리는 뉴런과 같은 유닛을 표현한다. 그리고 노드들은 간단히 계산이 어디서 발생하는지를 말한다. 노드들은 층이 다른 다른 노드들과 연결되며, 같은 층의 노드와는 연결되지 않는다.

같은 층간 커뮤니케이션이 없다 - 이것이 restricted Boltzmann machine의 제한이다.

각 노드는 입력을 처리하는 계산의 위치이다. 그리고 입력을 전송할 것인지에 대한 stochastic한 결정으로 시작한다. (stochastic은 "랜덤하게 결정된"이라는 의미이고 이경우, 입력을 수정하는 계수는 랜덤하게 초기화된다.)

각 visible 노드는 학습을 위해,  저-레벨 피쳐를 데이터셋의 item으로부터 취한다. 예를 들어, 흑백이미지의 데이터 셋에서, 각 visible 노드는 한 이미지에서 각 픽셀에 대한 하나의 픽셀값을 받는다. (MNIST images have 784 pixels, so neural nets processing them must have 784 input nodes on the visible layer.)

2층 넷에서 하나의 픽셀 값 x를 따라가보도록 하자. 은닉층<sup>hidden layer</sup>의 하나의 노드1에서,  x는 가중치와 곱해지고, bias와 더해진다. 이러한 두 연산의 결과는 노드의 출력(입력이 주어졌을 때, 노드를 통과하는 신호의 힘)을 생성하는 활성화 함수로 들어간다.

```java
activation f((weight w * input x) + bias b ) = output a
```
![enter image description here](http://deeplearning4j.org/img/input_path_RBM.png)
다음, 어떻게 몇개의 입력이 하나의 은닉노드에서 결합되는지 보자. 각 x는 분리된 가중치와 곱해진다. 곱은 합해지고, bias와 더해진다. 그 결과는 노드의 출력을 생성하는 활성화 함수를 통과한다.
![enter image description here](http://deeplearning4j.org/img/weighted_input_RBM.png)

모든 가시<sup>visible</sup> 노드로부터의 입력은 모든 은닉 노드로 전해진다. 하나의 RBM은 대칭적 이분 그래프<sup>symmetrical bipartite graph</sup>로 정의될 수 있다.

대칭은 각 가시 노드는 각 은닉노드와 서로 연결되어 있다는 의미이다. 이분은 그것이 두 부분 또는 층을 가졌다는 의미이고 그래프는 노드의 거미줄에 대한 수학적 용어이다.

각 가시 노드에서, 입력 $x$는 그와 대응하는 가중치 $w$와 곱해진다. 즉, 하나의 입력 x는 여기에서, 3개의 가중치를 가진다. 모두 12개의 가중치가 있다(4 input nodes x 3 hidden nodes). 두 층간 가중치는 항상 행렬을 생성한다. 각 행은 입력 노드와 같고 열은 출력노드와 같다. 

각 은닉 노드는 대응하는 가중치에 의해 곱해지는 4개의 입력을 받는다.  그러한 곱의 합은 항상 bias에 더해진다 (which forces at least some activations to happen). 그리고 그 결과는 하나의 은닉노드에 대해서 하나의 출력을 만드는 활성화 알고리즘에 전달된다.

![enter image description here](http://deeplearning4j.org/img/multiple_inputs_RBM.png)

이러한 두 개의 층이 딥 뉴럴 네트워크의 일부분이라면, 은닉 층 1번의 출력들은 은닉층 2번의 입력이 되고,  마지막 분류층에 도달할 때까지 원하는만큼의 은닉층에 대해서 반복된다. (For simple feed-forward movements, the RBM nodes function as an autoencoder and nothing more.)

![메롱](http://deeplearning4j.org/img/multiple_hidden_layers_RBM.png)

## Reconstructions
여기에서는, rbm이 데이터 재구성<sup>Reconstructions</sup>을 위해, 가시층과 하나의 은닉층을 오가며, 어떻게 비지도로 학습하는지 집중한다. 재구성 단계에서, 은닉층 1번의 활성은 backward pass로 입력이 된다.그것들은 x가 forward pass에서 조절됐던, 같은 가중치로 곱해진다.

그러한 곱셈의 합은 가시층의 각 가시유닛의 bias에 더해지고, 그러한 연산의 출력은 재구성이다. 즉 원래 입력의 근사이다. 이것은 다음 그림과 같이 표현될 수 있다.

![enter image description here](http://deeplearning4j.org/img/reconstruction_RBM.png)

RBM의 가중치는 랜덤하게 초기화되므로, 재구성과 원본입력 사이의 차이는 크다.재구성 오류를  `r`의 값들과 입력값들 사이에 차이로 볼 수 있다. 에러는 RBM의 가중치에 대해서 역전파될<sup>backpropagated</sup> 수 있다. 에러 최소값에 도달할때까지 반복적인 학습 과정을 수행한다.

역전파에 대한 자세한 설명은 [여기](http://deeplearning4j.org/neuralnet-overview.html#forward)

 forward pass에서, RBM은 노드 활성에 대한 예측을 하기위해, 입력을 사용한다. $p(a|x; w)$

그러나, backward pass에서, 활성은 입력이고 원본에 대한 재구성이 만들어진다. 하나의 RBM은 활성 $a$가 주어졌을 때, 입력 $x$의 확률을 추정한다.  활성 $a$는 forward pass에서 사용한 같은 계수의 가중치를 사용한다. 이 두번째 단계는 $p(x|a; w)$로 표현될 수 있다.

이러한 두개의 추정은 입력 $x$와 활성 $a$의 결합 확률 분포로 귀결된다. $p(x, a)$

 재구성은 회귀와는 다르다. 회귀는 많은 입력에 기반해서 연속적인 값을 추정한다. 그리고 분류와도 다르다. 분류는 주어진 입력 예제에 불연속적인 레이블을 추정한다.
 
재구성은 원본 입력의 확률 분포에 대한 추정을 한다. i.e. the values of many varied points at once. 이는 생성적<sup>generative</sup> 학습이라고 알려져 있다. [생성적 학습](http://cs229.stanford.edu/notes/cs229-notes2.pdf)은 분류에 의해 수행되는 소위 판별적<sup>discriminative</sup> 학습과는 구별된다. 판별적 학습은 데이터 점의 그룹사이의 선을 효율적으로 그려서 입력을 레이블로 매핑한다.

입력 데이터와 재구성 둘다 다른 모양의 정규분포이고, 일부분만 겹친다고 생각해보자.

추정된 확률분포와 입력의 정답 분포 사이의 거리를 측정하기 위해, RBMs는 [Kullback Leibler Divergence](https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-Divergence)를 사용한다.

 A thorough explanation of the math can be found on Wikipedia.
 수식에 대한 완전한 설명은 [Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)에서 볼 수 있다.
  
KL-Divergence은 두 곡선 밑 겹치지 않은 부분을 측정한다. RBM의 최적화 알고리즘은 이러한 영역을 최소화하려고 시도한다. 그래서 공유된 가중치는, 하나의 은닉층의 활성에 의해 곱해질때, 원본 입력의 가까운 근사를 만든다.

왼쪽은 재구성된 분포 $q$와 병렬배치된 원본 입력의 집합의 확률분포 $p$이다. 오른쪽은 그들의 차이들의 적분이다.
![enter image description here](http://deeplearning4j.org/img/KL_divergence_RBM.png)

그들이 만든 오류에 따라서 가중치를 반복적으로 조절함으로써, RBM은 원본 데이터를 근사하는 법을 학습한다. 가중치는 첫 가시층의 활성이 encode된 입력의 구조를 천천히 반영한다. 

학습 과정은 두 확률 분포가 단계별로 수렴하는 것으로 보인다. 
![enter image description here](http://deeplearning4j.org/img/KLD_update_RBM.png)

## Probability Distributions
잠시동안, 확률 분포에 대해서 이야기 해보자. 두 개의 주사위를 던지면, 결과의 확률분포는 다음과 같다.
![enter image description here](https://upload.wikimedia.org/wikipedia/commons/1/12/Dice_Distribution_%28bar%29.svg)

즉, 7s은 가장 확률이 높고,  and any formula attempting to predict the outcome of dice rolls needs to take that into account.

각 언어는 특정 문자를 다른 문자보다 많이 쓰기 때문에, 언어는 글자의 확률 분포로 특정된다. 영어에서는 문자 e와 a는 가장 많이 등장한다. 하지만 아이슬란드어에서는 가장 많이 나오는 문자는 r과 n이다. 영어에 기반한 가중치 셋으로 아이슬란드어를 재구성하는 것은 큰 발산으로 이끈다.

같은 방식으로, 이미지 데이터셋은 각 픽셀 값에 대한 고유의 확률 분포를 가진다.  Pixels values are distributed differently depending on whether the dataset includes MNIST’s handwritten numerals:
![enter image description here](http://deeplearning4j.org/img/mnist_render.png)
or the headshots found in Labeled Faces in the Wild:
![enter image description here](http://deeplearning4j.org/img/LFW_reconstruction.jpg)

RBM이 코끼리와 개 두개의 이미지만을 입력으로 받고, 각 동물에 대응하는 두 개의 출력 노드가 있다고 생각해보자.

node?  forward pass에서 RBM이 자신에서 묻는 질문은: 이러한 픽셀이 주어졌을때, 나의 가중치가 코끼리 노드 또는 강아지 노드에 대한 강한 신호를 보내야 하나?backward pass에서 RBM이 자신에서 묻는 질문은: elephant가 주어지면, 어떤 픽셀의 분포를 기대할 수 있나?

그것은 결합확률이다: RBM의 두 층 사이의 공유된 가중치로 표현되는 $a$가 주어질 때, $x$ 와 $x$가 주어질 때, $a$의 동시 확률

재구성을 학습하는 과정은, 어느정도는, 어떤 픽셀의 그룹이 이미지 집합에서 같이 등장할 것인가를 학습하는 것이다. 네트워크에서 깊은 은닉층의 노드가 만드는 활성은 유효한 co-occurrences를 표현한다. 예를 들어,  “nonlinear gray tube + big, floppy ears + wrinkles” might be one.

위의 두 이미지에서, RBM의 Deeplearning4j 구현에 의해서 학습된 재구성을 볼 수 있다. 이러한 재구성들은 RBM의 활성이 원본 데이터가 생긴 모습을 "생각"한 것이다. 제프리 힌튼은 이를 기계의 꿈이라고 칭하였다. 뉴럴넷 학습동안, 저런 시각화는 RBM이 실제로 학습되는지 확인할수 있는 유용한 휴리스틱이다. 제대로 학습되지 않는 다면, 밑에서 논의할 hyperparameters를 조정해야 한다.

마지막으로, RBM들은 두개의 bias를 가지고 있다. 이 것은 다른 autoencoders와 구별되는 점이다. 은닉 bias는 RBM이 forward pass에서 활성을 만들도록 돕는다.(since biases impose a floor so that at least some nodes fire no matter how sparse the data) 반면에, 가시 층의 bias는 RBM이 backward pass에서 재구성을 학습하도록 돕니다.

## Multiple Layers

Once this RBM learns the structure of the input data as it relates to the activations of the first hidden layer, then the data is passed one layer down the net. 

첫번째 은닉층은 이제 가시층의 역할을 떠맡는다. 활성은 지금 효과적으로 입력이 됐고, 두번째 은닉층의 노드에서 가중치와 곱해져서, 다른 활성을 만든다.

피쳐를 그룹핑하고 피쳐의 그룹을 다시 그룹핑함으로써 활성의 순서 집합을 만드는 과정은 피쳐 계층의 기초다. 이는 뉴럴넷이 좀 더 복잡하고 추상적인 데이터의 표현을 학습하게 한다.

각 새로운 은닉층에서,  그 층이 이전층의 입력을 근사할수 있게 가중치는 조절된다. 이는 greedy하고, layerwise한 비지도 사전학습<sup>pre-training</sup>이다. 네트워크의 가중치를 향상시킬 레이블을 요구하지 않는다.

보통, 알고리즘은 많은 데이터를 볼수록 좀더 나은 결과를 내놓는다.  그리고 이는 딥러닝 알고리즘이 우월한 이유 중에 하나이다.

그러한 가중치는 이미 데이터의 피쳐를 근사하므로, 이미 학습이 잘 되도록 잘 위치하여 있다. 그래서, 두번째 단계에서,deep-belief network으로 이미지를 더 잘 분류할 수 있다. 

RBMs은 많은 쓰임새가 있지만, 이후의 학습과 분류를 잘하게 만드는 가중치의 적절한 초기화는 그들의 주요 장점 중 하나이다. 그러한 점에서, 역전파<sup>backpropagation</sup>와 비슷한 효과가 있다: RBM과 역전파는 가중치가 데이터를 잘 모델링하도록 한다. You could say that pre-training and backprop are substitutable means to the same end.

restricted Boltzmann machines를 대칭 이분, 양방향 그래프로 다음 한그림에 그렸다.
![enter image description here](http://deeplearning4j.org/img/sym_bipartite_graph_RBM.png)
For those interested in studying the structure of RBMs in greater depth, they are one type of [directional, acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph).

## Code Sample: Initiating an RBM on Iris With DL4J
Note how, below, an RBM is simply created as a layer in a `NeuralNetConfiguration`, a parameter fed into a more general class. Likewise, the RBM object is used to store properties like the transforms applied to the visible and hidden layers, Gaussian and Rectified Linear transforms, respectively.

```java
public class RBMIrisExample {		

 private static Logger log = LoggerFactory.getLogger(RBMIrisExample.class);		

 public static void main(String[] args) throws IOException {		
     // Customizing params		
     Nd4j.MAX_SLICES_TO_PRINT = -1;		
     Nd4j.MAX_ELEMENTS_PER_SLICE = -1;		
     Nd4j.ENFORCE_NUMERICAL_STABILITY = true;		
     final int numRows = 4;		
     final int numColumns = 1;		
     int outputNum = 10;		
     int numSamples = 150;		
     int batchSize = 150;		
     int iterations = 100;		
     int seed = 123;		
     int listenerFreq = iterations/2;		

     log.info("Load data....");		
     DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);		
     // Loads data into generator and format consumable for NN		
     DataSet iris = iter.next();		

     iris.normalizeZeroMeanZeroUnitVariance();		

     log.info("Build model....");		
     NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().regularization(true)		
             .miniBatch(true)		
             // Gaussian for visible; Rectified for hidden		
             // Set contrastive divergence to 1		
             .layer(new RBM.Builder().l2(1e-1).l1(1e-3)		
                     .nIn(numRows * numColumns) // Input nodes		
                     .nOut(outputNum) // Output nodes		
                     .activation("relu") // Activation function type		
                     .weightInit(WeightInit.RELU) // Weight initialization		
                     .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)		
                     .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)		
                     .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)		
                     .build())		
             .seed(seed) // Locks in weight initialization for tuning		
             .iterations(iterations)		
             .learningRate(1e-3) // Backprop step size		
             // Speed of modifying learning rate		
             .optimizationAlgo(OptimizationAlgorithm.LBFGS)		
                     // ^^ Calculates gradients		
             .build();		
     Layer model = LayerFactories.getFactory(conf.getLayer()).create(conf);		
     model.setListeners(new ScoreIterationListener(listenerFreq));		

     log.info("Evaluate weights....");		
     INDArray w = model.getParam(DefaultParamInitializer.WEIGHT_KEY);		
     log.info("Weights: " + w);		
     log.info("Scaling the dataset");		
     iris.scale();		
     log.info("Train model....");		
     for(int i = 0; i < 20; i++) {		
         log.info("Epoch "+i+":");		
         model.fit(iris.getFeatureMatrix());		
     }		
 }		
 // A single layer learns features unsupervised.
}
```
This is an example of an RBM processing the Iris flower dataset.

## Parameters & k
The variable k is the number of times you run contrastive divergence. Contrastive divergence is the method used to calculate the gradient (the slope representing the relationship between a network’s weights and its error), without which no learning can occur.

Each time contrastive divergence is run, it’s a sample of the Markov Chain composing the restricted Boltzmann machine. A typical value is 1.

In the above example, you can see how RBMs can be created as layers with a more general MultiLayerConfiguration. After each dot you’ll find an additional parameter that affects the structure and performance of a deep neural net. Most of those parameters are defined on this site.

weightInit, or weightInitialization represents the starting value of the coefficients that amplify or mute the input signal coming into each node. Proper weight initialization can save you a lot of training time, because training a net is nothing more than adjusting the coefficients to transmit the best signals, which allow the net to classify accurately.

activationFunction refers to one of a set of functions that determine the threshold(s) at each node above which a signal is passed through the node, and below which it is blocked. If a node passes the signal through, it is “activated.”

optimizationAlgo refers to the manner by which a neural net minimizes error, or finds a locus of least error, as it adjusts its coefficients step by step. LBFGS, an acronym whose letters each refer to the last names of its multiple inventors, is an optimization algorithm that makes use of second-order derivatives to calculate the slope of gradient along which coefficients are adjusted.

regularization methods such as l2 help fight overfitting in neural nets. Regularization essentially punishes large coefficients, since large coefficients by definition mean the net has learned to pin its results to a few heavily weighted inputs. Overly strong weights can make it difficult to generalize a net’s model when exposed to new data.

VisibleUnit/HiddenUnit refers to the layers of a neural net. The VisibleUnit, or layer, is the layer of nodes where input goes in, and the HiddenUnit is the layer where those inputs are recombined in more complex features. Both units have their own so-called transforms, in this case Gaussian for the visible and Rectified Linear for the hidden, which map the signal coming out of their respective layers onto a new space.

lossFunction is the way you measure error, or the difference between your net’s guesses and the correct labels contained in the test set. Here we use SQUARED_ERROR, which makes all errors positive so they can be summed and backpropagated.

learningRate, like momentum, affects how much the neural net adjusts the coefficients on each iteration as it corrects for error. These two parameters help determine the size of the steps the net takes down the gradient towards a local optimum. A large learning rate will make the net learn fast, and maybe overshoot the optimum. A small learning rate will slow down the learning, which can be inefficient.





