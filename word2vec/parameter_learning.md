## word2vec Parameter Learning Explained
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