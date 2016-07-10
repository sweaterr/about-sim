User interactions with search engines provide many cues that can be leveraged to improve the relevance of search results through personalization. 
사용자 상호작용을 가진 검색 엔진은 개인화를 통한 검색 결과 연관성을 향상시키는 많은 힌트를 제공한다.

The context information (history of queries, clicked documents, etc.) provides strong signals about users’ search intent, which can be used to personalize the search experience and improve a web search engine. 
상황 정보(쿼리 히스토리, 클릭한 문서 등)은 유저의 검색 의도에 대한 강한 신호를 제공한다. 이 검색의도는 검색 경험을 개인화하는데 사용할 수 있고, 웹 검색엔진을 발전시킬 수 있다.

We demonstrate how to generate the semantic features from in-session contextual information with deep learning models, and incorporate these semantic features into the current ranking model to re-rank the results. 
우리는 딥러닝 모델을 가지고 세션 상황 정보로부터 시만틱 피쳐를 어떻게 생성했는지 입증했다. 그리고 이 시멕틱 피쳐를 랭킹 결과를 리랭킹하는데 사용했다.

We evaluate our approach using a large, real-world search log data from a major commercial web search engine, and the experimental results show our approach can significantly improve the performance of the search engine. 
이 접근법을 크고, 상용 검색엔진의 실세계 검색 로그 데이터로 평가했고, 실험적 결과는 우리의 접근법이 검색엔진의 성능을 유의미하게 향상시킴을 보였다 

Furthermore, we also find that the domain-specific, click-based features can effectively decrease the unsatisfied clicks for the current ranking model to improve the search experience.
더 나가가, 도메인-특정, 클릭-기반 피쳐가 검색 경험을 증가시키는 현재 랭킹 모델에 대한 불만족 클릭을 감소시킴을 알아냈다.
