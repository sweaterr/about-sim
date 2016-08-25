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
= \frac{\exp(\mathbf{v'}_{w_j}^T\mathbf{\mathbf{v}^T_{w_I}})}{\sum_{j'=1}^{V}{\exp(\mathbf{v'}_{w_{j'}}^T\mathbf{v}^T_{w_I}})} 
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
#### Reference
* [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
