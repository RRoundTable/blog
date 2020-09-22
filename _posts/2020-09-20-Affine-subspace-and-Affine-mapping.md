---
title: "Affine subspace and Affine mapping 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Affine subspace

<img src="https://upload.wikimedia.org/wikipedia/commons/9/95/Affine_space_R3.png" style="zoom:30%;" />



- $V$: vector space
- $x_0 \in V$
- $U \subset V$

$$
L = x_0 + U = \{x_0 + u: u \in U \}
= \{v \in V \mid  \exists u \in U: v = x_0 + u \} \subset V
$$

L은 affine subspace이며, V의 linear manifold로 불리기도 한다. U는 direction 또는 direction space로 불린다. $x_0$는 supprot point이며, 추후에 hyperplane으로 불리기도 한다.

다음과 같이 두 개의 affine subspace가 있다고 생각해보자. 

- $L = x_0 + U$
- $\tilde{L} = \tilde{x_0} + \tilde{U}$

다음과 같은 명제가 성립한다.

$L \subset \tilde{L}$ if and only if $U \subset \tilde{U}$ and $x_0 - \tilde{x_0} \in \tilde{U}$

또한 affine subspace는 다음과 같이, U의 basis의 linear combination으로도 나타낼 수 있다.

- $(b_1, b_2, \cdots, c_n)$ : basis of U

$$
x = x_0 + b_1 * \alpha_1 + \cdots + b_k \alpha_k
$$

### 차원에 따른 affine subspace 명칭

- One-dimension: line
- Two-dimension: plane
- more dimension: hyperplane\

### Inhomorgeneous system of linear equations and affine subspaces

- $A \in R ^{mxn}$, $x \in R^m$, $x \neq 0$

아래의 linear equation의 해는 empty set 혹은 affine subspace(n - rk(A))이다. 
$$
A \lambda = x
$$

$$
\lambda_1 b_1 + \cdots + \lambda_n b_n = x \text{  where  } (\lambda_1, \cdots, \lambda_n) \neq (0, \cdots, 0)
$$

의 해는 hyperplane이다.





## Affine mapping

- two vector space V, W

- Linear mapping $\Phi$: $V \rightarrow W$ and $a \in W$ 

- $$
  \Phi:V \rightarrow W \\
  x \rightarrow a + \Phi(x)
  $$

- 모든 affine mapping은 linear mapping 과 translation으로 표현할 수 있다.

- Affine mapping과 Affine mapping의 합성함수는 Affine mapping이다.

- affine mappings keep the geometric structure invariant. They also pre- serve the dimension and parallelism.







