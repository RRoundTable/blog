---

title: "Matrix Approximation 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['math', 'linear algebra']
layout: post
---



## Matrix Approximation

[SVD 관련글](https://rroundtable.github.io/blog/math/linear%20algebra/2020/11/17/singular-value-decomposition.html)에서 $A$라는 matrix가 $A = U \Sigma V^T \in R^{m \times n}$로 분해되는 것을 살펴보았다. 앞으로 살펴볼 내용은 모든 $U, V$의 모든 vector를 사용하는 것이 아니라 일부를 사용하여 matrix approximation하는 방법에 대해서 다룰 것이다. 이렇게 근사하게 되면 상대적으로 낮은 computation으로 matrix를 재현할 수 있다.



### Example

![]({{ site.baseurl }}/images/2020-12-05-matrix-approximation/example.png)

위의 이미지는 어떤 이미지 데이터를 가지고 rank-1 matrix로 분해한 것이다.

### rank-1 matrix

$$
A_i := v_iu_i^T
$$

- $A_i \in R^{m \times n}$

- outer product of column vector of U and V

  

### Matrix consisted with outer product matrix

$$
A = \sum_{i=1}^{r}\sigma^i u_iv_i^T = \sum_{i=1}^r \sigma_iA_i
$$

- $A \in R^{m \times n}$: rank가 r이다.
- $A_i$: outer product matrix
- $\sigma_i$: singular value, weight의 역할을 함.
- $u_iv_j^T \forall i \neq j$: $\Sigma$는 diagonal matrix이므로 0으로 없어진다.
- $i > r$: 은 모두 sigular value가 0이다.



### Rank-k approximation(k < r)

$$
\hat{A}(k) :=\sum_{i=1}^{k}\sigma_i u_iv_i^T = \sum_{i=1}^k \sigma_iA_i
$$

- $rk(\hat{A}(k))=k$

![]({{ site.baseurl }}/images/2020-12-05-matrix-approximation/rank.png)

위의 그림처럼 rank가 r에 가까워질수록 원본 이미지와 유사해진다.



### Definition 4.23 (spectral norm of a matrix)

For $x \in R^n \backslash \left\{ 0\right\}$, spectral norm of matrix $A \in R^{m \times n}$ is defined as
$$
\rVert A \rVert_2 := \max_{x} \frac{\rVert Ax \rVert_2}{\rVert x \rVert_2}
$$


### Theorem 4.24

spectral norm of A is the largest sigular value of A.



### Theorem 4.25: Eckart-Young theorem

- $rank(A) = r, A \in R^{m \times n}$
- $\hat{A}(k) :=\sum_{i=1}^{k}\sigma_i u_iv_i^T $
- $rank(B) = k, B \in R^{m \times n}$
- $k \leqslant r$

$$
\hat{A}(k) = \arg \min_{rk(B)=k} \rVert A- B\rVert_2 \\
\rVert A- \hat{A}(k)\rVert_2 = \sigma_{i+1}
$$



Eckart-Young theorem은 approximated matrix $\hat{A}(k)$가 얼만큼의 오차를 가지는지 표현하며, 위의 식을 보면 그 오차를 최소한으로 하는 matrix를 구한다.

이렇게 구한 $\hat{A}(k)$를 A에 빼보면, 아래와 같이 표현된다. 따라서 theorem 4.24를 활용하면, spectral norm은 $\sigma_{k+1}$이다.
$$
A- \hat{A}(k) = \sum_{i=k+1}^r \sigma_i u_iv_i^T
$$


$\hat{A}(k) = \arg \min_{rk(B)=k} \rVert A- B\rVert_2 $에 대해서 더 자세히 살펴보자.

아래와 같은 matrix B가 있다고 가정해보자. 즉 $\hat{A}(k)$보다 더 최적의 matrix B가 있다고 가정하는 것이다.

- $rank(B) \leqslant k$

$$
\rVert A - B \rVert_2 < \rVert A- \hat{A}(k)\rVert_2
$$



그렇다면, 최소한 $n - k$의 null space $Z \subset R^n$가 존재하며 이는 $Bx = 0 \forall x \in Z$를 의미한다.

- $x \in Z$

$$
\rVert Ax\rVert_2 = \rVert (A - B)x\rVert_2
$$



Cauchy-Schwartz inequality를 사용하면, 아래와 같이 전개가 가능하다.
$$
\rVert Ax\rVert_2 \leqslant \rVert (A - B)\rVert_2 \rVert x\rVert_2  \leqslant \sigma_{k+1}\rVert x\rVert_2
$$
**Cauchy-Schwartz inequality**
$$
\rVert Ax \rVert \leqslant \rVert A\rVert_2\rVert x\rVert_2
$$




하지만, 이는 rank-nullity theorem과 상충된다. 왜냐하면, 아래의 조건을 만족하는 subspace의 차원은 k + 1이기 때문이다. null space의 차원과 더 하면, 총 n + 1차원이된다.
$$
\rVert Ax \rVert_2 \le \sigma_{k+1} x
$$

- $v_1, \cdots, \v_{k+1}$: right sigular vector



따라서, rank k를 가지면서 A를 근사할 수 있는 matrix는 아래와 같다.
$$
\hat{A}(k) = \arg \min_{rk(B)=k} \rVert A- B\rVert_2
$$






****