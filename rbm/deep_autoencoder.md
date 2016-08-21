[원문](http://deeplearning4j.org/deepautoencoder)
A deep autoencoder is composed of two, symmetrical deep-belief networks that typically have four or five shallow layers representing the encoding half of the net, and second set of four or five layers that make up the decoding half.
딥오토인코더는 두개의 대칭적 딥-빌리프 네트워크로 이루어져 있다 / 일반적으로 4에서 5개의 얕은 층을 가진 / 반은 인코딩을 표현하고 / 두번째 반은 디코딩을 한다

The layers are restricted Boltzmann machines, the building blocks of deep-belief networks, with several peculiarities that we’ll discuss below. 
층은 RBM이다 / 딥 빌리프 네트워크의 블럭을 세우는 / 몇 개의 특성으로 / 밑에서 논의할 / 
Here’s a simplified schema of a deep autoencoder’s structure, which we’ll explain below.
간단한 스키마가 있다 / 딥 오토인코더의 구조의 / 밑에서 설명할 
![enter image description here](http://deeplearning4j.org/img/deep_autoencoder.png)
Processing the benchmark dataset MNIST, a deep autoencoder would use binary transformations after each RBM. 
벤치마크 MNIST 데이터셋를 처리하는 / 딥 오토인코더는 이진 변환을 사용할 것이다 / 각 RBM 후에
Deep autoencoders can also be used for other types of datasets with real-valued data, on which you would use Gaussian rectified transformations for the RBMs instead.
딥 오토인코더는 사용될 수 있다 / 다른 타입의 데이터셋에 대해서 / 실수값의 /  Gaussian rectified transformations를 사용할 / RBM 대신에 

### Encoding
Let’s sketch out an example encoder:
인코더를 스케치하자
```
 784 (input) ----> 1000 ----> 500 ----> 250 ----> 100 -----> 30
```
If, say, the input fed to the network is 784 pixels (the square of the 28x28 pixel images in the MNIST dataset), then the first layer of the deep autoencoder should have 1000 parameters; i.e. slightly larger.
입력은 / 네트워크에 들어가는 /  784 픽셀이라면 /  (the square of the 28x28 pixel images in the MNIST dataset) / 딥 오토인코더의 첫번째 층은 1000개의 파라메터를 가진다 / 즉 살짝 크다.

This may seem counterintuitive, because having more parameters than input is a good way to overfit a neural network.
이는 직관에 반한다 / 왜냐하면 / 입력보다 많은 파라메터를 갖는 것은 좋은 방법이다 / 뉴럴넷을 오버핏 시키는 

In this case, expanding the parameters, and in a sense expanding the features of the input itself, will make the eventual decoding of the autoencoded data possible.
이 경우 / 파라메터를 확장하는 것은 / 그리고 어느정도 / 입력 자체의 피쳐를 확장시키는 것은 / 결국엔 오토인코딩된 데이터의 디코딩을 가능하게 한다.

This is due to the representational capacity of sigmoid-belief units, a form of transformation used with each layer. 
이는 표현적 용량 때문이다 / 시그모이드-빌리프 유닛의 / 변환의 형태인 / 각 층에서 사용하는 
Sigmoid belief units can’t represent as much as information and variance as real-valued data. 
시그모이드 빌리프 유닛은 표현할 수 없다 / 정보와 변동을 / 실수값 정도의 
The expanded first layer is a way of compensating for that.
확장된 첫번째 층은 방법이다 / 그것을 압축하는 

The layers will be 1000, 500, 250, 100 nodes wide, respectively, until the end, where the net produces a vector 30 numbers long. 
그 층은 1000, 500, 250, 100 노드가 있다 / 각각 / 끝날때에 / 넷은 30 개의 숫자의 벡터를 생산한다 / 
This 30-number vector is the last layer of the first half of the deep autoencoder, the pretraining half, and it is the product of a normal RBM, rather than an classification output layer such as Softmax or logistic regression, as you would normally see at the end of a deep-belief network.
이 30숫자 벡터는 마지막 층이다 / 딥 오토인코더의 첫번째 반절의 / 사전 학습 반절 / 그리고 그것은 일반적 RBM의 곱이다 / 분류 출력보다는 / 소프트 맥스나 로지스틱 리그레션 같은 / 당신이 일반적으로 딥 빌리프 네트워크 끝에서 보는 

### 디코딩
Those 30 numbers are an encoded version of the 28x28 pixel image. 
그러한 30개숫자들은 28x28 픽셀 이미지의 인코딩버전이다.
The second half of a deep autoencoder actually learns how to decode the condensed vector, which becomes the input as it makes its way back.
딥 오토인코더의 두번째 반절은 압축된 벡터를 어떻게 디코드하는지 학습한다 / 입력이 되는 / 다시 되돌아 가는 

The decoding half of a deep autoencoder is a feed-forward net with layers 100, 250, 500 and 1000 nodes wide, respectively. 
딥오토인코더의 디코딩 반절은 100, 250, 500 and 1000 노드 층을 가진  피드포워드 넷이다.
Those layers initially have the same weights as their counterparts in the pretraining net, except that the weights are transposed; i.e. they are not initialized randomly.)
그러한 층은 초기에 같은 가중치를 가진다 / 반대 부분으로 / 사전학습 넷에서 / 가중치가 transpose됐다는 점만 다르게 / 즉 랜덤하게 초기화 되지 않는다.
```
	784 (output) <---- 1000 <---- 500 <---- 250 <---- 30
```
The decoding half of a deep autoencoder is the part that learns to reconstruct the image. 
딥오토인코더의 디코딩 반절은 부분이다 / 이미지를 재구성하는 것을 배우는
It does so with a second feed-forward net which also conducts back propagation. 
그것은 두번째 피드포워드 넷도 마찬가지이다 / 또한 역전파를 수행하는
The back propagation happens through reconstruction entropy.
역전파는 재구성 엔트로피에 대해서 일어난다.

## Training Nuances
At the stage of the decoder’s backpropagation, the learning rate should be lowered, or made slower:
디코더의 역전파의 단계에서, 학습률은 작아야한다
somewhere between 1e-3 and 1e-6, depending on whether you’re handling binary or continuous data, respectively.

## Use Cases
### Image Search
As we mentioned above, deep autoencoders are capable of compressing images into 30-number vectors.
위에서 언급했는듯, 딥 오토인코더들은 이미지를 30차원 벡터로 압측할 수 있다.

Image search, therefore, becomes a matter of uploading an image, which the search engine will then compress to 30 numbers, and compare that vector to all the others in its index.
이미지 검색에서, 그러므로 / 이미지를 업로딩하는 문제가 되었다 / 검색엔진은 30개의 숫자로 압축할 것이고 / 그 벡터를 다른 것과 비교한다 / 색인의

Vectors containing similar numbers will be returned for the search query, and translated into their matching image.
비슷한 숫자들를 담는 벡터들은 / 리턴된다 / 검색 쿼리에 대해서 / 그리고 매칭되는 이미지로 변환된다.

###Data Compression
A more general case of image compression is data compression. 
좀 더 일반적인 경우는 / 이미지 압축의 / 데이터 압축이다.
Deep autoencoders are useful for [semantic hashing](https://www.cs.utoronto.ca/~rsalakhu/papers/semantic_final.pdf), as discussed in this paper by Geoff Hinton.
딥 오토인코더는 / 유용하다 / 시멘틱 해슁에 / 제프리 힌튼 논문에서 논의된 

###Topic Modeling & Information Retrieval (IR)
Deep autoencoders are useful in topic modeling, or statistically modeling abstract topics that are distributed across a collection of documents.
딥 오토 인코더는 유용하다 / 토픽모델링에 / 또는 통계적인 추상적 토픽을 모델링 하는데 / 문서의 콜레션을 넘어 분산되어 있는

This, in turn, is an important step in question-answer systems like Watson.
이는 차례차례 중요한 단계이다 / QA 시스템에서 / 왓슨과 같은

In brief, each document in a collection is converted to a Bag-of-Words (i.e. a set of word counts) and those word counts are scaled to decimals between 0 and 1, which may be thought of as the probability of a word occurring in the doc.
요약하면, 콜렉션의 각 문서는 / 백오브워드로 변환된다 / 즉 단어 카운트의 집합 / 그리고 그러한 단어 카운트는 0에서 1사이로 스케일링된다 / 확률로서 생각할 수 있다 / 단어의 / 하나의 문서에 등장하는

The scaled word counts are then fed into a deep-belief network, a stack of restricted Boltzmann machines, which themselves are just a subset of feedforward-backprop autoencoders. 
스케일링된 단어 카운트는 딥빌리프넷에 들어간다 / RBM의 스택인 / 그자체로 피드포워드 역전파 딥오토인코더의 부분집합인
Those deep-belief networks, or DBNs, compress each document to a set of 10 numbers through a series of sigmoid transforms that map it onto the feature space.
그런한 딥 빌리프 네트워크 또는 DBN은 각 문서를 10개 숫자의 집합으로 압측한다 / 시그모이드 변화의 연속을 통해서 / 그것을 피쳐 공간에 매핑하는 

Each document’s number set, or vector, is then introduced to the same vector space, and its distance from every other document-vector measured. 
각 문서의 숫자 집합 즉 벡터는 같은 벡터스페이스에 들어간다 / 그리고 모든 다른 문서-벡터와의 거리가 / 측정된다
Roughly speaking, nearby document-vectors fall under the same topic.
러프하게 말하면 / 가까운 문서벡터는 같은 토픽에 떨어진다

For example, one document could be the “question” and others could be the “answers,” a match the software would make using vector-space measurements.
예를 들어 / 하나의 문서는 "질문"이 될수 있다 / 그리고 다른 것은 "대답"이 될수 있다 / 소프트웨에가 할 수 있는 / 벡터공간 측정으로 / 매치되는 

## Code Sample
A deep auto encoder can be built by extending Deeplearning4j’s MultiLayerNetwork class.

The code would look something like this:

final int numRows = 28; final int numColumns = 28; int seed = 123; int numSamples = MnistDataFetcher.NUM_EXAMPLES; int batchSize = 1000; int iterations = 1; int listenerFreq = iterations/5;

```java
log.info("Load data....");
    DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

    log.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(10)
            .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) 
            
            //encoding stops
            .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) 	
            
            //decoding starts
            .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numRows*numColumns).build())
            .pretrain(true).backprop(true)
            .build();

     MultiLayerNetwork model = new MultiLayerNetwork(conf);
     model.init();

     model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

     log.info("Train model....");
     while(iter.hasNext()) {
        DataSet next = iter.next();
        model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
```
To construct a deep autoencoder, please make sure you have the most recent version of Deeplearning4j and its examples, which are at 0.4.x.

For questions about Deep Autoencoders, contact us on Gitter.