# GloVe: Global Vectors for Word Representation

## abstract
* Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. 
* 단어의 벡터 공간 표현을 학습하는 최근 방법들은 미세한 의미론적 문법적 규칙을 잡아내는데 성공했다 / 벡터 연산을 사용해서 / 그러나 이러한 규칙의 기원은 아직도 불투명하다
* We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. 
* 우리는 분석했다/ 그리고 모델 성질을 명시적으로 만들었다 / 모델 성질은 그러한 규칙성이 단어 벡터에 나타나도록 하는데 필요하다. 
* The result is a new global log-bilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. 
* 그 결과가 새로운 글로벌 log-bilinear 회귀 모델이다 / 두 주요 모델 패밀리의 장점을 조합한 / 글로벌 행렬 분해와 로컬 문맥 윈도우 모델이라는
* Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. 
* 우리의 모델은 효율적으로 통계적 정보를 써먹는다 / 단지 0이 아닌 원소들만 훈련함으로써 / 단어-단어 동시등장 행렬에서 / 전체 희귀 행렬 또는 개별 문맥 윈도우 보다 / 큰 문서 집합에서
* The model produces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task.
* 그 모델은 벡터공간을 생성한다 / 의미있는 부분구조를 갖는 / 75%의 성능이 증명된 / 최근 단어 유사 문제에 대해서
* It also outperforms related models on similarity tasks and named entity recognition.
* 다른 모델도 이겼다 / 유사도 문제와 인명 인식 문제에 대해서
 
