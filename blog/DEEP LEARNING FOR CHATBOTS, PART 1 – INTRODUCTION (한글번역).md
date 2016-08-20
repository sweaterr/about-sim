## DEEP LEARNING FOR CHATBOTS, PART 1 – INTRODUCTION (한글번역)

@[published]


다음 [포스트](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)를, 좀 더 자세히 읽으려는 공부목적으로 번역해보았습니다. 

Conversational Agents 또는 Dialog Systems이라 불리는 Chatbots은 핫토픽이다. Microsoft는 chatbots에 [큰 배팅](http://www.bloomberg.com/features/2016-microsoft-future-ai-chatbots/)을 했고, Facebook, Apple, Google, WeChat, Slack같은 회사들도 뛰어들었다. 소비자가 Operator 같은 앱 또는 Chatfuel같은 x.ai, bot platforms를 통해  서비스와 교류할 수 있게 하는 시도를 하는 스타트업의 새로운 물결이 일고 있다. Microsoft는 최근 자신들만의 [봇 개발 프레임웍](https://dev.botframework.com/)을 출시했다.

많은 회사들이 자연스러운, 인간과 구별이 안되는 봇을 개발하기를 원한다. 그리고 많은이들이  NLP 기술과 딥러닝 기술을 사용해서 이것을 가능하게 할 수 있다고 주장하고 있다. 그러나 모든 AI의 과대광고가 섞여있으면, 픽션과 구별된 사실을 말하기가 어렵다.

## 모델 분류

##### RETRIEVAL-BASED VS. GENERATIVE MODELS

**검색-기반 모델(쉬움)**은 입력과 문맥에 기반해서 적절한 응답을 고르는, 선-정의된 응답과 어떤 종류의 휴리스틱의 저장소이다. 휴리스틱은 룰-기반 표현 매칭처럼 간단하거나, 머신러닝 분류기의 앙상블처럼 복잡하다. 이 시스템은 새로운 텍스트를 생성하지 않는다. 단지 고정된 집합에서 응답을 고른다.

**생성모델<sup>Generative models</sup> (어려움)**은 선-정의된 응답에 의지하지 않는다. 이 모델들은 밑바닥부터 새로운 응답을 생성한다. 생성 모델은 일반적으로 기계 번역 기술에 기반한다. 그러나, 하나의 언어를 다른언어로 번역하는 대신에,  입력으로부터 다른 출력을 "번역"한다.

![enter image description here](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/nct-seq2seq.png)

두 접근법 둘다 장단점이 확실하다. 손으로 만든 응답의 저장소 때문에, 검색-기반 방식은 문법적 실수를 하지 않는다.
그러나, 적절한 선-정의된 응답이 존재하지 않는 겪지 않은 케이스에 대해서는 응답하지 않는다. 같은 이유로, 이 모델은 대화중 언급된 이름과 같은 문맥적 개체 정보를 다시 언급할 수 없다. Generative models은 더 똑똑하다. 입력에서의 개체들을 다시 언급할수 있고, 마치 사람과 얘기하고 있는 인상을 줄 수 있다. 그러나, 이 모델은 훈련하기 어렵고 문법적 실수를 할 가능성이 있고 일반적으로 엄청난 량의 훈련데이터를 요구한다.

딥러닝 기술은 검색-기반 또는 생성모델에 둘다 쓰일수 있지만, 연구는 생성방향으로 움직이고 있다. Sequence to Sequence은 딥러닝 구조는 텍스트를 생성하는데 잘 맞고 연구자들은 이 영역에 대한 큰 진보를 희망한다. 그러나, 아직 우리 생각대로 잘 동작하는 생성모델의 초기단계에 있다. 제품 시스템은 아직까지 검색-기반이 더 많다.

##### LONG VS. SHORT CONVERSATIONS

대화가 길어질수록, 더 자동화하기 어렵다. 스펙트럼의 한쪽은 Short-Text Conversations (easier)이다. Short-Text Conversations (easier)은 단일 입력에 단일 응답을 생성하는게 목적이다. 예를 들어, 당신은 사용자들로부터 특정 질문을 받고, 적절한 대답을 할 수도 있다. 그 후,  서로 대화를 주고 받으며, 긴대화(어려움)가 있을 수 있다. 고객 지원 대화는 보통 다수의 질문이 있는 대화의 쓰레드<sup>thread</sup>이다.

##### OPEN DOMAIN VS. CLOSED DOMAIN

열린도메인<sup>open domain(harder)</sup>  환경에서는 사용자가 대화를 어디에서든 할 수 있다. 잘-정의된 목표와 의도가 없을수도 있다. 트위터나 레딧같은 소셜미디어 사이트에서의 대화는 일반적으로 열린도메인이다 - 모든 방향으로 대화를 이끌수 있다. 수많은 토픽과 특정 양의 세계에 대한 지식이 합리적인 응답을 위해 필요하다는 사실은 이 문제를 어려운 문제로 만든다. 가능한 입력과 출력이 정해진 닫힌도메인<sup>close domain</sup> 환경에서는 시스템이 매우 특정 목표를 성취하기를 시도하기 때문에 다소 한계가 있다. Technical Customer Support 또는 Shopping Assistants는 이 닫힌도메인 문제의 예이다. 이 시스템은 정치에 대해서 이야기할 능력이 필요하지 않고, 특정문제만 효율적으로 처리하면 된다. 확실히, 사용자는 자기가 원하는 어디서든 대화할 수 있지만, 시스템은 모든 경우를 다룰 필요는 없고 사용자가 그걸 기대하지도 않는다.

### COMMON CHALLENGES
대화형 에이전트를 만드는데 명백한, 그리고 다소 명백하지 않은 문제들이 있다. 전부 활발한 연구 영역이다.


##### 문맥 합치기<sup>INCORPORATING CONTEXT</sup>

분별있는 응답 시스템을 만드는 것은 linguistic context 과 physical context을 둘 다 합쳐야 한다. 긴 대화에서, 사람들은 무엇을 말했는지, 어떤 정보가 교환됐는지 계속 기억하고 있는다. 그것이 linguistic context의 예이다. 가장 흔한 접근법은 대화를 벡터로 바꾸는 [embed](https://en.wikipedia.org/wiki/Word_embedding) 이지만, 긴 대화에 대해서 하는 것은 어렵다. [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](http://arxiv.org/abs/1507.04808) 과 [Attention with Intention for a Neural Network Conversation Model](Attention%20with%20Intention%20for%20a%20Neural%20Network%20Conversation%20Model) 에서의 실험은 둘다 그 방향으로 하고 있다. 시간, 위치, 사용자정보같은 문맥정보를 합쳐야 할 수도 있다.

##### 성향 일관성<sup>COHERENT PERSONALITY</sup>
When generating responses the agent should ideally produce consistent answers to semantically identical inputs. 
응답을 생성할떄, 에이전트는 의미적으로 동일한 입력에 대해서 일관성 있는 대답을 해야한다. 예를 들어, "몇살이예요?", "나이가 어떻게 되세요?"라는 질문에 같은 대답을 원한다. 이는 간단해 보이지만, 바뀌지 않는 지식 또는 성향을 모델에 합치는 것은 매우 어려운 문제이다. 많은 시스템은 linguistic하게 그럴듯한 응답을 생성하도록 학습하지만, 의미적으로 일관성있게 생성하도록 학습하지는 않는다. 보통 다수의 다른 유저들로부터 많은 데이터를 학습하기 때문이다. [A Persona-Based Neural Conversation Model](http://arxiv.org/abs/1603.06155)는 명시적으로 성향을 모델링하는 방향에 대한 첫걸음이다.


##### EVALUATION OF MODELS
대화형 에이전트를 평가하는 이상적인 방법은 임무를 충분히 수행했는지 안했는지 측정하는 것이다. 예를 들어 대화가 주어였을떄 고객 지원 문제를 풀기같은 것이 있다. 그러나, 그러한 레이블은 얻기 힘들다. 왜냐하면, 인간 검수자와 평가가 필요하기 때문이다. 때때로, 열린도메인의 경우처럼 잘 정의된 목적이 없다.

기계 번역과 텍스트매칭에서 [BLEU](https://en.wikipedia.org/wiki/BLEU) 같은 많이 쓰는 메트릭<sup>metric</sup>은 잘 맞지 않는다. 왜냐하면 분별있는 응답은 완전히 다른 단어 또는 어절이 포함될 수 있기 때문이다.

In fact, in [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](http://arxiv.org/abs/1603.08023) researchers find that none of the commonly used metrics really correlate with human judgment.
사실, [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](http://arxiv.org/abs/1603.08023) 에서 연구자들은 어떤 흔히 쓰는 메트릭도 human judgment와 상관관계가 없다는 것을 알아냈다.

##### INTENTION AND DIVERSITY

A common problem with generative systems is that they tend to produce generic responses like “That’s great!” or “I don’t know” that work for a lot of input cases.
생성 시스템의 공통적인 문제는 “That’s great!” 또는 “I don’t know”와 같은 많은 입력에 잘 동작하는 일반적인 응답을 하는 경향이 많다는 것이다. 구글의 [Smart Reply](http://googleresearch.blogspot.com/2015/11/computer-respond-to-this-email.html) 는 모든 응답에 사랑한다고 말하는 경향이 있다. 어떻게 이 시스템이 데이터, 목적/알고리즘 측면에서 훈련됐는지에 대한 부분적 결과이다. [어떤 연구자들은 다양한 목적 함수를 통해  인공적으로 다양성을 증진하는 것을 시도했었다.](http://arxiv.org/abs/1510.03055)  그러나 인간은 일반적으로 입력에 대해서 특정 응답을 하고 의도를 담는다.  생성 시스템(특히, 열린도메인시스템)은 특정 의도를 가지도록 훈련되지 않았으므로, 이러한 종류의 다양성이 부족하다.

### HOW WELL DOES IT ACTUALLY WORK?
최신 연구가 주어졌으므로, 이러한 시스템이 실제로 잘 동작하는지 알아보자. 분류를 다시 고려해 보자. 검색-기반 열린 도메인 시스템은 명백히 불가능하다. 왜냐하면 모든 경우에 대해서 손으로 응답을 만들수 없기 때문이다. 생성 열린도메인 시스템은 거의 Artificial General Intelligence (AGI)이다. 왜냐하면 모든 가능한 경우를 다루어야하기 때문이다. This leaves us with problems in restricted domains where both generative and retrieval based methods are appropriate.  대화가 길어지면, 문맥이 더 중요해지고, 문제도 더 어려워진다.

바이두의 최고 과학자인 Andrew Ng의 최근 인터뷰에 잘 드러나있다.

> 딥러닝의 대부분의 가치는 데이터를 얻을 수 있는 좁은 도메인이다. 딥러닝이 하지 못하는 한가지 예가 있다: 의미있는 대화 나누기이다. 대화를 신중하게 고르면, 매우 의미있는 대화를 나누고 있는것처럼 보이지만, 실제로 시도해 보면, 빠르게  탈선하는 데모가 있다.

많은 회사들이 그들의 대화를 인간 노동자들에게 외주를 맡김으로써 시작하고  일단 충분한 데이터를 모았다면 노동자가 대화를 자동화해줄것이라고 약속한다. 예를 들어, 우버로 전화하는 챗 인터페이스 같은 좁은 도메인에서 동작한다면, 그 말은 맞다. 판매 메일같은 좀 더 오픈도메인은 현재 우리가 할 수 있는 영역 밖이다.  그러나, 응답을 제안하고 고쳐줌으로써 인간 노동자를 도와주는 식으로 사용할 수 있다. 


production systems에서 문법적 실수는 매우 위험하고, 유저를 떠나게 할 것이다. 대부분의 시스템은 문법적 에러와 공격적인 응답에 대해서 자유로운 검색기반 방법이 제일좋을 것이다. 회사가 다소 매우 큰 데이터를 손에 얻었다면, 그 후엔 생성 모델이 가능하다 - 그러나 Microsoft’s Tay가 했던 탈선행위를 방지하기 위해서 다른 기술이 뒷받침되어야 할것이다.


### UPCOMING & READING LIST

We’ll get into the technical details of how to implement retrieval-based and generative conversational models using Deep Learning in the next post, but if you’re interested in looking at some of the research then the following papers are a good starting point:

* [Neural Responding Machine for Short-Text Conversation (2015-03)](http://arxiv.org/abs/1503.02364)
* [A Neural Conversational Model (2015-06)](http://arxiv.org/abs/1506.05869)
* [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses (2015-06)](A%20Neural%20Network%20Approach%20to%20Context-Sensitive%20Generation%20of%20Conversational%20Responses%20%282015-06%29)
* [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems (2015-06)](The%20Ubuntu%20Dialogue%20Corpus:%20A%20Large%20Dataset%20for%20Research%20in%20Unstructured%20Multi-Turn%20Dialogue%20Systems%20%282015-06%29)
* [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models (2015-07)](Building%20End-To-End%20Dialogue%20Systems%20Using%20Generative%20Hierarchical%20Neural%20Network%20Models%20%282015-07%29)
* [A Diversity-Promoting Objective Function for Neural Conversation Models (2015-10)](A%20Diversity-Promoting%20Objective%20Function%20for%20Neural%20Conversation%20Models%20%282015-10%29)
* [Attention with Intention for a Neural Network Conversation Model (2015-10)](Attention%20with%20Intention%20for%20a%20Neural%20Network%20Conversation%20Model%20%282015-10%29)
* [Improved Deep Learning Baselines for Ubuntu Corpus Dialogs (2015-10)](http://arxiv.org/abs/1510.03753)
* [A Survey of Available Corpora for Building Data-Driven Dialogue Systems (2015-12)](A%20Survey%20of%20Available%20Corpora%20for%20Building%20Data-Driven%20Dialogue%20Systems%20%282015-12%29)
* [Incorporating Copying Mechanism in Sequence-to-Sequence Learning (2016-03)](http://arxiv.org/abs/1603.06393)
* [A Persona-Based Neural Conversation Model (2016-03)](http://arxiv.org/abs/1603.06155)
* [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation (2016-03)](http://arxiv.org/abs/1603.08023)













