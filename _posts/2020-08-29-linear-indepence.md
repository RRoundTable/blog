---
title: "linear indepence 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Linear combination

Linear combination 아래의 수식과 같이 나타낼 수 있다.
$$
x_1, x_2,\cdots, x_n \in V \\
\lambda_1, \lambda_2, \cdots, \lambda_k \in R
$$

$$
v = \lambda_1 x_1 + \lambda_2 x_2 + \cdots + \lambda_n x_n
$$



## Linear indepence

아래와 같이 k개의 vector가 있을때,
$$
x_1, x_2,\cdots, x_k \in V
$$

$$
\lambda_1 x_1 + \lambda_2 x_2 + \cdots + \lambda_k x_k = 0
$$

위의 식을 만족하는 경우는 다음과 같은 경우를 생각해볼 수 있다.

1. $\lambda_1 = \cdots = \lambda_k = 0$ 
2. linear combination을 통해서, 다른 vector를 재현할 수 있는 경우.



만약, $\lambda_1 x_1 + \lambda_2 x_2 + \cdots + \lambda_k x_k = 0$을 만들기 위해서 오직 1번만 가능하다면, linear independence라고 하며, 그 외의 경우는 linear dependent하다고 정의한다.



## More

다음과 같이 $b_1, b_2, \cdots, b_k$의 linear independent vectors가 있고, m개의 linear combination이 있다면 다음과 같이 표현할 수 있다.
$$
x_1 = \sum_{i=1}^k \lambda_{i1}b_i\\
\vdots \\
x_m = \sum_{i=1}^k \lambda_{im}b_i\\
$$
그리고, $x_j$는 다음과 같이 나타낼 수 있다.
$$
x_j = B\lambda_j
$$




그렇다면, $x_1, \cdots, x_m$이 linear independent한 조건을 알아보자.
$$
\sum_{j=1}^m \psi_j x_j = \sum_{j=1}^m \psi_j B \lambda_j = B \sum_{j=1}^m \psi_j \lambda_j = 0
$$


위의 식의 해가 $\psi_1 = \cdots = \psi_m = 0$  인 경우 뿐이라면, linear independent하다.  그리고 이는 $\lambda$가 linear independent하다면, X도 linear independent하다는 것을 의미한다.



아래식의 해가  $\psi_1 = \cdots = \psi_m = 0$ 만 있어야 하므로, $\lambda$는 linear independent해야한다.
$$
\sum_{j=1}^m\psi_j \lambda_j = 0
$$


## Affine indepedent

Affine independent를 정의하기 위해서는 affine set에 대해서 간략히 정리하고 가겠다. 쉽게 표현하면, 점(point), 직선(line), 평면(plane), 초평면(hyperplane)과 같이 선형적 특성이 있으면서 경계가 없는 집합을 말한다. 어떤 집합이 affine set이라고 말할 수 있으려면 집합에 속한 임의의 두 점으로 직선을 만들어서 그 직선이 집합에 포함되는지를 보면 된다. 
$$
\theta x_1 + (1-\theta)x_2 \in C \text{  with  } \theta \in R
$$


아래의 그림에서 P2를 보면, 위의 조건이 성립함을 알 수 있으며, 이를 affine subspace라고 부른다. 하지만, vector subspace는 0을 지나지 않으므로 아니다. 반면에 P1은 vector subspace이다. $\vec{a}, \vec{b}$는 p2에 있지만, 둘의 distance vector는 P1상에 위치한다. 정리하면, affine subspace는 vector subspace의 일반화 버전이라고 생각할 수 있다. 그리고 이를 수식으로 아래와 같이 표현할 수 있다.

<img src="https://upload.wikimedia.org/wikipedia/commons/9/95/Affine_space_R3.png" style="zoom:30%;" />

- C: affine set
- V: vector subspace

$$
V = C - x_0 = \{ x - x_0 \phantom{1} | \phantom{1} x \in C \}
$$



V가 vector subpace임을 보여보자. vector subpace 연산에 대해서 닫혀있는지 확인해보자.
$$
v1, v2 \in V , \ \alpha, \beta \in R ,\  x_0 \in C
$$
$\alpha v_1 + \beta v_2 + x_0 $ 가 affine set이라면, $alpha v_1 + \beta v_2 $는 vector subspace이다.
$$
\alpha v_1 + \beta v_2 + x_0 = \alpha (v_1 + x_0) + \beta (v_2 + x_0) + (1 - \alpha - \beta) x_0 \in C
$$


affine independent는 다음과 같이 정의된다.

아래와 같이 vector, scalar가 정의되었을때, 
$$
v_1, \cdots, v_k \in V, \ \sum_{i=1}^k \theta_i = 0 \\
$$
아래와 같이 성립한다면, affine independent하다.
$$
\sum_{i=1}^k \theta_i v_i = 0 \text{ if and only if  } \theta_1 = \cdots = \theta_k = 0
$$


이를 조금 다른 표현으로 나타내보면,

- \${ x - x_0 \phantom{1} | \phantom{1} x \in C \}$ 이 linear independent하다.

- 아래의 vector가 linear independent하다.

$$
\begin{vmatrix} 1 \\v_1 \end{vmatrix}, \cdots, \begin{vmatrix} 1 \\v_k \end{vmatrix}
$$



- reference.
  - https://wikidocs.net/21467
  - https://wikidocs.net/17412

