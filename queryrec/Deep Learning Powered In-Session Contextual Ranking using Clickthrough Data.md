## abstract 
* User interactions with search engines provide many cues that can be leveraged to improve the relevance of search results through personalization. 
사용자 상호작용을 가진 검색 엔진은 개인화를 통한 검색 결과 연관성을 향상시키는 많은 힌트를 제공한다.
*  The context information (history of queries, clicked documents, etc.) provides strong signals about users’ search intent, which can be used to personalize the search experience and improve a web search engine. 
상황 정보(쿼리 히스토리, 클릭한 문서 등)은 유저의 검색 의도에 대한 강한 신호를 제공한다. 이 검색의도는 검색 경험을 개인화하는데 사용할 수 있고, 웹 검색엔진을 발전시킬 수 있다.
* We demonstrate how to generate the semantic features from in-session contextual information with deep learning models, and incorporate these semantic features into the current ranking model to re-rank the results. 
우리는 딥러닝 모델을 가지고 세션 상황 정보로부터 시만틱 피쳐를 어떻게 생성했는지 입증했다. 그리고 이 시멕틱 피쳐를 랭킹 결과를 리랭킹하는데 사용했다.
* We evaluate our approach using a large, real-world search log data from a major commercial web search engine, and the experimental results show our approach can significantly improve the performance of the search engine. 
이 접근법을 크고, 상용 검색엔진의 실세계 검색 로그 데이터로 평가했고, 실험적 결과는 우리의 접근법이 검색엔진의 성능을 유의미하게 향상시킴을 보였다 
* Furthermore, we also find that the domain-specific, click-based features can effectively decrease the unsatisfied clicks for the current ranking model to improve the search experience.
더 나가가, 도메인-특정, 클릭-기반 피쳐가 검색 경험을 증가시키는 현재 랭킹 모델에 대한 불만족 클릭을 감소시킴을 알아냈다.

## 1 Introduction
* Search engines have become a ubiquitous part of everyday life: users issue queries, examine ordered
results, click on certain links, spend some time on pages, reformulate their queries, and launch their
search again. 

* During this interaction with the search engine, users provide some valuable implicit
or indirect judgment to the search engine, partially revealing cues about their search intent. 

* In recent years, both in academia and industry, there is growing interest in investigating how users’
contextual information, including users’ interests, search history, clickthrough data etc., can be used
to improve various search related tasks, such as query reformulation, query suggestion, shopping
recommendation, ranking etc. 

* Generally two categories of context information can be mined from these search logs: Long-term context refers to some long-term (across the whole search history), stable information about users, such as users’ general interests; Short-term context is the immediate information surrounding users’ current search needs in a short time span, e.g.

* users’ search history in a session, their clicked documents, their dwelling time over the clicked documents etc. 

* Both kind of contexts provide valuable information for personalization.

-- 

* Currently, some commercial search engines take into account some coarse-grained signals derived
from historical queries. 
* For example, one existing whole-page feature in the major commercial search engine considered here is “AveQueryOverlap,” which computes the token overlap between pairs of consecutive queries in the current session. 
* Obviously this feature is coarse, lacking some semantic information derived from historical queries and current query. 
* Another example is whether it has some domain features from previous queries, e.g. all the queries issued by one user in the last month; and it also has some distribution information about users’ click behaviors to reflect users’ domain preference, but not strictly associated to their search satisfaction.
* Here is an example.
* EXAMPLE 1: Table 1 shows two queries (2nd and 6th queries) in one session from real log data.

![image](https://cloud.githubusercontent.com/assets/1518919/16714278/23463462-46f9-11e6-9385-2e6f65e855cd.png)

* The difference between query 2 and query 6 is query specification by adding the term of “wiki” in
query 6. 
쿼리 2와 쿼리 6의 차이는 쿼리의 구체화이다 / 쿼리 6 안에 위키라는 단어를 추가한

* In query 2, the corresponding Wikipedia page was clicked as a unsatisfied click (red color
in the table), and in this case it is good to demote the Wikipedia page from the first position of the
6th query since it was examined as a unsatisfied page in query 2, while, in reality, the wikia page in
the second position of the 6th query is a satisfied click (blue color in the table) from users log. 

* Thus, it is better to promote the position of wikia page in the ranking result. 
* We will address this problem at the end of this paper.

--
* In this work, we focus on the short-term context, employ latent semantic deep learning models
to generate the semantic features from in-session context, and investigate how to use the shortterm
signals to improve ranking. 
* We incorporate some semantic, expressive signals (e.g. topic, domain) derived from in-session context into the ranking, to re-rank the results. 
* Here, a session can be considered as a sequence of interactions for the same information need within a short period.
* Automatically detecting the boundary of the sessions is beyond this paper, which has been studied
previously [10].
* We assume that the session boundaries are known. 
* The specific contributions of this paper include:
 * Beyond preference, we correlate the users clicks with their satisfaction and propose a set of
fine-grained semantic features to explore the relations between in-session history queries,
clicked URLs to the current query and URLs.
 * Employ the deep learning models [2, 3] to measure the semantic similarity for the above semantic features.
 * Besides semantic features, we also find that domain click-based features could effectively decrease the unsatisfied click rate to improve the search experience of users.
 

## 3 Approach
* In this section, we formalize the problem, and then describe our approach. 
* We briefly summarize how to derive the fine-grained semantic features between the current query, URLs and their context information, apply the deep learning models to calculate their semantic similarity, and evaluate the effectiveness of contextual information for ranking using the real log data from a major commercial search engine.

### 3.1 Problem Definition
* In a session, a user may interact with the search engine several times. 
* During interactions, the user would modify his/her query based on the past search experience. 
* 상호 작용동안, 사용자는 과거의 검색 경험을 기반으로 그들의 쿼리를 바꾼다.
* Therefore for the current query Q(except for the first query in a search session), there is a query history Q1, Q2, . . ., Qm associated with it. 
* 그러므로, 현재 쿼리 Q에 대해서(검색 세션의 첫번째 쿼리를 제외한) 그와 연관된 쿼리 이력 Q1, Q2, ..., Qm이 있다.
* For each historical query, there are some clicked URLs with multi-fold information, i.e.
dwelling time on that page. 
* 각 이전 쿼리에 대해서, 그 페이지에 대한 체류시간 같은 다양한 정보를 가진 클릭된 url이 있다.
* Hence, the context of current query often contains the queries asked before Q as well as the answers (URLs) to those queries that were clicked on or skipped by the user.
* 그러므로, 현재 쿼리에 대한 문맥은 자주 
For a query Qt in a session at time t, we constrain the context of Qt to only the query Qi
asked before Qt in the same session and the answers to Qi clicked on or skipped by the user. In
practice, the real world log data is usually noisy, contains a lot of “quickback” queries (identical
queries issued by the user in a short time, or consecutively), here we would recommend to ignore
these quickback queries. Figure 1 illustrates the problem.
