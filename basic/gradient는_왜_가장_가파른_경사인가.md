gradient의 각 원소는 함수가 각 축에서 얼마나 빨리 변하냐를 알려준다. 그렇다면, 어떤 임의의 방향에 대해서 함수는 얼마나 빨리 변하는지 당연히 궁금할 것이다.  $\mathbf{v}$를 단위벡터라 하자. gradient를 이 방향으로 다음과 같이 사영시킬수 있다. 
 $$grad(f(a)) \cdot \mathbf{v}$$
이것은 directional derivative의 정의이다 그렇다면, 저 변화율이 최대가 되는 방향벡터\mathbf{v}은 어느방향인가?
$$
\text{grad}( f(a))\cdot \mathbf v = |\text{grad}( f(a))|| \mathbf v|\text{cos}(\theta)
$$
$\mathbf v$는 단위 벡터이기 때문에 $|\text{grad}( f(a))|\text{cos}(\theta)$를 얻을 수 있고 $\cos(\theta)=1$일 때 전체값이 최대가 되므로, $\text{grad}( f(a))$이 $\mathbf v$과 방향이 같을 때, 변화율이 최대가 된다. 즉 gradient가 가장 큰 변화율을 가진 방향이다.

