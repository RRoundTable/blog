---

title: "inner product 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---

inner product는 기하학에서 length, angle, distance 등과 같은 개념을 파생시킵니다.



## 3.2.1 Dot product

inner product의 대표적인 예입니다. 주의할 점은, inner product는 dot product를 포함하지 동치의 관계가 아닙니다.
$$
x^T y = \sum_{i=1}^n x_iy_i
$$

## 3.2.2 General Inner Product

그렇다면, 조금 더 general한 inner product에 대해서 알아보겠습니다. 

**bilinear mapping**

general inner product를 이해하기 위해서 먼저 bilinear mapping에 대해서 알아보고 가겠습니다.

- $x, y , z \in V$
- $\lambda, \psi in R$

$$
\Psi(\lambda x + \psi y, z) = \lambda \Psi(x, z) + \psi\Psi(y, z)
$$

$$
\Psi(x, \lambda y + \psi z) = \lambda\Psi(x, y) + \psi \Psi(x, z)
$$



- $V$: vector space
- $\Psi: V \times V \rightarrow R$: bilinear mapping

$\Psi$ symmetric:$\Psi(x, y) = \Psi(y, x) \forall x,y in V$

$\Psi$ positive definite:  $\forall x in V / {0}: \Psi(x, x) > 0, \Psi(0, 0) = 0$

**inner product**

- $V$: vector space
- $\Psi: V x V \rightarrow R$: bilinear mapping

inner product: positive definite, symmetric bilinear mapping

$(V, <\cdot, \cdot>)$은 inner product space이다. 만약 inner product를 dot product를 사용한다면, 그것을 유클리디안 공간이라고 부른다.



### Example 3.3 **Inner Product That Is Not the Dot Product**

$$
<x, y> := x_1y_1 -(x_1y_2+  x_2y_1) + 2x_2y_2
$$

위의 예시는 bilinear mapping이며, 또한 positive definite, symmetric하다. 따라서, inner product라고 할 수 있다.



## 3.2.3 Symmetric, Positive Definite Matrices

Symmetric, positive definite matrix는 machine learning에서 매우 중요한 역할을 한다. 또한, inner product를 통해서 정의된다. 추후에 다루겠지만,  section 12.4 kernel에서 매우 중요한 개념이다.

n-dimensional vector space V와 inner product $<\cdot, \cdot>$을 가정하자. 

그리고 ordered basis $B = (b_1, b_2, \cdots, b_n)$이 있다고 생각해보자. $x, y \in V$는 ordered basis의 linear combination을 통해서 정의할 수 표현할 수 있다. 그리고 이를 수식으로 표현하면 다음과 같다.

- $\lambda_i, \psi_i \in R$

$$
x = \sum_{i=1}^n \psi_i b_i \in V
$$

$$
x = \sum_{i=1}^n \lambda_i b_i \in V
$$



그리고 $x, y$의 inner product는 다음과 같이 표현할 수 있다.
$$
<x, y> = <\sum_{i=1}^n \psi_i b_i, 	\sum_{i=1}^n \lambda_i b_i> = \sum_{i=1}^n \sum_{j=1}^n \psi_i <b_i, b_j>\lambda_j = \hat{x}^T A \hat{y}
$$

- $\hat{x}, \hat{y}$: the coordinates of x and y with respect to the basis B, 각각 $\psi_i, \lambda_i$의 값

위의 식을 통해서, A를 통해서 inner product가 unique하게 결정되는 것을 알 수 있다. 또한 inner product의 symmetry한 특성은 A또한 symmetry metrix라는 것을 의미한다. 게다가, inner product의 positive definites는 A가 다음과 같은 특성을 가지는 것을 알 수 있다.
$$
\forall x\in V \setminus \{0\}, x^TAx > 0
$$
A가 symmetric, positive definite matrix라면, 아래의 식처럼 inner product를 나타낼 수 있다.
$$
<x, y> = \hat{x}^T A \hat{y}
$$




**Definition: symmetric, positive definite matrix**

symmetric matrix $A$가 다음과 같은 조건을 만족하면, symmetric, positive definite matrix이다.
$$
\forall x\in V \setminus \{0\}, x^TAx > 0
$$

### Theroem 3.5

$$
<x, y> = \hat{x}^T A \hat{y} \text{ is inner product} \iff A \text{  is  symmetric positive definite matrix}
$$



Symmetric, positive definite matrix A는 다음과 같은 특징을 가진다.

- null space는 오직 0 vector 뿐이다.
  - $x^T A x > 0$
- A의 대각성분은 모두 양수이다.
  - $e_i^T A e_i > 0$
  - scalar를 vector를 이용해서 표현하면 어떻게 할 수 있을까?