## word2vec Parameter Learning Explained
![image](https://cloud.githubusercontent.com/assets/1518919/17956311/dde3f00a-6ac3-11e6-9516-2503e3fe3ada.png)
$$
\mathbf{h} =\mathbf{W}^T\mathbf{x}= \mathbf{W}^T_{k,\cdot}=\mathbf{v}^T_{w_I}
$$
$\mathbf{W}$는 크기 $V \times N$ 행렬
$V$는 단어집 크기
$N$는 임베딩 벡터 크기
$\mathbf{x}$는 $V$크기의 one-hot 벡터
$\mathbf{W'}$는 크기 $N \times V$ 행렬
$$
u_j = \mathbf{v'}_{w_j}^T\mathbf{h}
$$

$$
p(w_j|w_I)=y_j=\frac{\exp(u_j)}{\sum_{j'=1}^{V}{\exp(u_{j'})}} \\
= \frac{\exp(\mathbf{v'}_{w_j}^T\mathbf{\mathbf{v}_{w_I}})}{\sum_{j'=1}^{V}{\exp(\mathbf{v'}_{w_{j'}}^T\mathbf{v}_{w_I}})} 
$$
## Update equation for hidden→output weights
$$
\log p(w_O|w_I) = \log y_{j^*} \\
= u_{j^*} - \log \sum_{j'=1}^{V} \exp(u_{j'}) = -E
$$

$$
\frac{\partial E}{\partial u_j} =\frac{\partial \log \sum_{j'=1}^{V} \exp(u_{j'}) - u_{j^*}}{\partial u_j}  \\
= y_j - t_j = e_j
$$
$\frac{\partial {u_{j^*}}}{\partial {u_j}}$은 $j$가 target word의 index일 때 1이 되고, 다른 단어의 index일 때는 0이다. 그러므로 $t_j$로 표시할 수 있다.

$$
\frac{\partial E}{\partial w_{ij}'} = \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial w_{ij}'} = e_j \cdot h_i
$$

$$
w_{ij}^{'new} = w_{ij}^{'(old)} - \eta \cdot e_j \cdot h_i
 $$
또는, $j = 1,2,...,V $에 대해서 
$$
\mathbf{v'}_{w_j}^{(new)}=\mathbf{v'}_{w_j}^{(old)}-\eta \cdot e_j \cdot \mathbf{h} 		
$$

##Update equation for input→hidden weights
$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{V}\frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial h_i} = \sum_{j=1}^{V}e_j \cdot w'_{ij} = \mathbf{EH_i}
$$
$\mathbf{E}.shape = 1 \times V $
$\mathbf{H_i}.shape = V \times 1 $
$$
h_i = \sum_{k=1}^{V}{x_k} \cdot w_{ki} 
$$

$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial w_{ki}} = \mathbf{EH_i} \cdot x_k
$$

$$
\frac{\partial E}{\partial \mathbf{W}} = \mathbf x \otimes \mathbf {EH} = \mathbf {x}\mathbf {E}\mathbf {H}^T
$$

$\mathbf{x}.shape=V \times 1$
$\mathbf{E}.shape=1 \times V$
$\mathbf{H}^T.shape=V \times N$

$$
\mathbf{v}_{w_I}^{(new)}=\mathbf{v}_{w_I}^{(old)}-\eta\mathbf{E}\mathbf{H}^{T} 
$$
$\mathbf{E}\mathbf{H}^{T}$는 단어집에서 출력벡터의 예측에러($e_j = y_j - t_j$)로 가중치된 모든 단어의 합이다. $\mathbf{v}_{w_I}^{(new)}=\mathbf{v}_{w_I}^{(old)} -\eta\mathbf{E}\mathbf{H}^{T} $은 문맥 단어의 입력벡터에 단어집의 모든 출력 벡터의 한부분을 더하는 것으로 볼 수 있다. 출력층에서 단어 $w_j$가 출력단어가 될 확률이 과추정된다면($y_j > t_j$), 입력 벡터 $w_I$는 $w_j$의 출력벡터에서 멀어질 것이다. $y_j < t_j$ 일 경우, $y_j = t_j$일 경우는, 직접 생각해보자.

문서로부터 만들어진 문맥-타겟 페어를 돌면서 모델 파라메터는 반복적으로 업데이트된다. 단어 $w$의 출력벡터는 $w$와 같이 등장하는 이웃의 입력벡터에 의해서 앞뒤로 끌어당겨진다. 마찬가지로 입력벡터도 많은 출력벡터에 의해서 끌어당겨진다.

### 멀티단어 문맥
CBOW는 multi-word context setting이다. 
$$
\mathbf{h}  = \frac{1}{C}\mathbf{W}^T(\mathbf{x_1} + \mathbf{x_2} + ... + \mathbf{x_C}) \\
  = \frac{1}{C}(\mathbf{v}_{w_1} + \mathbf{v}_{w_2} + ... + \mathbf{v}_{w_C})
$$

또는, $j = 1,2,...,V $에 대해서 
$$
\mathbf{v'}_{w_j}^{(new)}=\mathbf{v'}_{w_j}^{(old)}-\eta \cdot e_j \cdot \mathbf{h} 		
$$
또는, $c = 1,2,...,C $에 대해서 
$$
\mathbf{v}_{w_{I,c}}^{(new)}=\mathbf{v}_{w_{I,c}}^{(old)}-\frac{1}{C}\eta\mathbf{E}\mathbf{H}^{T} 
$$

Skip-gram은 

##Optimizing Computational Efficiency
입력벡터를 학습하는 것은 싸다. 하지만 출력벡터를 학습하는 것은 비싸다. 출력벡터를 학습하는 것은 다음과 같이 모든 단어를 돌면서 해야한다.
$j = 1,2,...,V $에 대해서 
$$
\mathbf{v'}_{w_j}^{(new)}=\mathbf{v'}_{w_j}^{(old)}-\eta \cdot e_j \cdot \mathbf{h} 		
$$

### Hierarchical Softmax
![image](https://cloud.githubusercontent.com/assets/1518919/17993122/1d4016d8-6b88-11e6-8324-b1bd2ebf3297.png)
$n(w, j)$은 루트에서 단어 $w$까지의 경로에서 $j$번째 유닛을 의미한다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027232/b57c3456-6c99-11e6-95db-077ce1de2de4.png)
$ch(n)$은 유닛의 왼쪽 자식 
$\mathbf{v'}_{n(w,j)}$은 안쪽 유닛의 n(w,j) 벡터 표현(출력 벡터)
$\mathbf{h}$는 은닉층의 출력값(skip-gram은 $\mathbf{v_{w_I}}$이고 CBOW에서는 $\mathbf{h} = \frac{1}{C}\sum_{c=1}^{C}\mathbf{v}_{w_c}$
![image](https://cloud.githubusercontent.com/assets/1518919/18027464/d991ef7a-6c9e-11e6-8d83-ea0603762d03.png)
예제로 식을 이해해 보자. 그림4를 보면, $w_2$가 출력 단어가 될 확률을 계산하기를 원한다고 하자. 이 확률을 뿌리부터 시작해서 잎 유닛으로 끝나는 랜덤워크의 확률로 정의할 수 있다. 각 내부 유닛에서(뿌리를 포함한), 왼쪽으로 갈 확률과 오른쪽으로 갈 확률의 할당이 필요하다. 내부 유닛 $n$에서 왼쪽으로 갈 확률을 다음과 같이 정의할 수 있다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027478/5f578caa-6c9f-11e6-83cc-45e3bdadd996.png)
그렇다면, 오른쪽으로 갈 확률은 다음과 같다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027485/9d24d9c0-6c9f-11e6-88b6-ba56fd610639.png)
그림4에서 뿌리에서 w2까지 경로를 따라가면, $w2$이 출력 단어가 될 확률을 다음과 같이 계산할 수 있다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027491/c50e23ec-6c9f-11e6-8434-07e3e4d2270f.png)
다음을 증명하는 것은 어렵지 않다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027497/10c9302e-6ca0-11e6-9365-588c222d39e0.png)
이는 계층적 소프트맥스를 모든 단어에 대한 잘 정의된 다항 분포로 만든다.

이제, 계층적 소프트맥스의 내부 유닛에 대한, 업데이트 식을 유도해보자. 간소함을 위해서 한-단어 문맥을 본다. CBOW나 skip-gram으로 확장하는 것은 쉽다. 수식의 간소함을 위해서, 중의성 없는 다음과 같은 단축 수식을 정의한다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027520/28b98e58-6ca1-11e6-8852-620af83737d1.png)
![image](https://cloud.githubusercontent.com/assets/1518919/18027526/3a4d3124-6ca1-11e6-9dca-64e7d57ea494.png)
하나의 훈련 예제에 대해서, 에러 함수는 다음과 같이 정의된다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027532/6e58aa7a-6ca1-11e6-88f6-aab363f3623d.png)
log로 인해서 곱셈은 덧셈으로 바뀌었다.
$\mathbf{v'}_j \mathbf{h}$에 대해서 E의 도함수를 취하면 

![image](https://cloud.githubusercontent.com/assets/1518919/18027544/d195b100-6ca1-11e6-85a8-56e568f1ad73.png)
유닛$n(w, j)$의 벡터 표현에 대해서 E의 도함수를 취하면, 
![image](https://cloud.githubusercontent.com/assets/1518919/18027641/29d01dcc-6ca4-11e6-99f1-50f81edb3b0f.png)
이제 다음과 같은 업데이트 식이 나온다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027643/3eda0142-6ca4-11e6-8c59-6e0f2492747a.png)
$j = 1, 2, ..., L(w) -1$에 대해서 적용되어야 한다. (계산 가볍). $\sigma(\mathbf{v'}_j^T\mathbf{h}) - t_j$를 내부 유닛 $n(w, j)$에 대한 에러라고 이해할 수 있다. 각 내부 유닛에 대한 "임무"는 랜덤워크에서 왼쪽 자식을 따라야 하는지 오른쪽 자식을 따라야 하는지 예측하는 것이다. $t_j=1$는 정답이 왼쪽을 따르는 것임을 의미하고, $t_j=0$는 오른쪽 자식을 따른 것임을 의미한다. $\sigma(\mathbf{v'}_j^T\mathbf{h})$는 예측 결과이다. 하나의 트레이닝 예제에 대해서, 내부 유닉의 예측이 정답과 근사했다면, $\mathbf{v'}_j$는 매우 조금 움직일 것이다.




## Negative Sampling
네가티브 샘플링의 아이디어는 계층 소프트맥스보다 직관적이다: 너무 많은 출력벡터를 가짐의 어려움을 극복하기 위해, 그 이부분만을 업데이트한다.

출력단어(즉, 정답, 양의 샘플)은 우리의 샘플안에 유지되어야 하고 업데이트 되어야 한다. 그리고 몇 단어를 음의 샘플로 필요하다. 확률 분포는 샘플링 과정에 필요하다. 그리고 임의로 고를 수 있다. 이 분포를 노이즈 분포라고 부른다. 그리고 $P_n(w)$라고 표시한다. 좋은 분포를 경험적으로 결정할 수도 있다.

word2vec에서, 잘 정의된 사후 다항 분포를 생성하는 음의 샘플링의 형태를 사용하는 대신에, 저자들은 다음과같은 간소화된 훈련 목적이 고품질의 단어 임베딩을 만들수 있다고 주장했다.
$$
E = -\log \sigma ( \mathbf{v'}_{w_O} \mathbf{h})  - \sum_{w_j \in \mathcal{W_{neg}}}{ \log\sigma ( \mathbf{v'}_{w_j} \mathbf{h}}  ) 
$$

$w_O$은 출력 단어(양의 샘플)
$\mathbf{v'}_{w_O}$은 출력 벡터
$\mathbf{h}$는 은닉층의 출력값
$\mathbf{h}=\frac{1}{C}\sum_{c=1}^{C}\mathbf{v}_{w_c}$은  CBOW
$\mathbf{h}=\mathbf{v}_{w_I}$은  Skip-gram
$\mathcal{W_{neg}} = \{w_j|j = 1, ..., K \}$은 $P_n(w)$로 샘플링된 단어의 집합

![image](https://cloud.githubusercontent.com/assets/1518919/18026948/259bdfc4-6c91-11e6-9107-b6679d5479c1.png)
$\mathbf{v'}^{T}_{w_j} \mathbf{h}$는 context 벡터 $\mathbf{h}$가 단어 $w_j$의 출력벡터와 비슷한 정도이다(내적이기 때문에).  두 벡터가 비슷하면, $\sigma ( \mathbf{v'}_{w_j} \mathbf{h}) $가 커지고, $t_j$가 1이어야 에러가 작다.


![image](https://cloud.githubusercontent.com/assets/1518919/18026951/43274ff6-6c91-11e6-8207-2875b5d83544.png)

![image](https://cloud.githubusercontent.com/assets/1518919/18027114/5a4b4710-6c95-11e6-90d0-dce127296281.png)
모든 단어가 아니라 $w_j \in \{w_O \} \cup \mathcal{W_{neg}}$에 있는 단어만 적용하면 되므로, 계산 효율적이다.

은닉층의 에러를 역전파하고, 단어의 입력 벡터를 업데이트 하기 위해서, 은닉층에 대한 E의 도함수가 필요하다.
![image](https://cloud.githubusercontent.com/assets/1518919/18027134/3c7237ca-6c96-11e6-90d8-ff77c6c34401.png)

![image](https://cloud.githubusercontent.com/assets/1518919/18027136/4a60b17c-6c96-11e6-94bf-ee028b6f8484.png)
(60)을 (22)에 대압히면 CBOW 모델에 대한 입력벡터 업데이트 식을 얻을 수 있다.
skip-gram 모델에 대해서는 EH를 context의 각 단어에 대해서 계산해서 더해야한다.
#### Reference
* [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
