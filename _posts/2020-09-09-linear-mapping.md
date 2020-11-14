---
title: "linear mapping 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Linear Mapping

vector space $V, W$ 에 대하여, mapping $\psi: V \rightarrow \Phi W$ 이 있을때, 다음과 같은 조건을 만족하면 linear mapping이라고 한다.


$$
\forall x, y \in V \ \forall \lambda, \psi \in R: \Phi(\lambda x + \psi y) = \lambda \Phi(x) + \psi \Phi(y)
$$


아래에서 다루겠지만, 이런 linear mapping은 matrix로 표현이 가능하다.



### Injective, Surjective, Bijective 

![](https://ds055uzetaobb.cloudfront.net/brioche/uploads/EkswlzPrzb-examp.svg?width=300)

- injective는 위의 그림처럼 정의역과 치역의 mapping이 1대1관계이다. 하지만 공역과는 1대1 대응은 아니다.
- Surjective는 정의역이 모든 공역에 mapping된다.
- bijective는 정의역이 모든 공역에 1대1로 mapping되는 것을 의미한다. 

특히, bijective mapping이 중요하다. 왜냐하면, bijective 조건이 만족해야 역함수를 정의할 수 있기 때문이다.



### Isomorphism, Endomorphism, Automorphism

- isomorphism: $\Phi: V \rightarrow W$, linear and bijective

- endomorphism: $\Phi: V \rightarrow V$, linear

- automorphism: $\Phi: V \rightarrow V$, linear, bijective

  

## Example 2.19 (Homomorphism)

$$
\Phi: R^2 \rightarrow \mathbb{C}, \Phi(x) = x_1 + ix_2
$$

$$
\Phi(\begin{bmatrix} x_1 \\ x_2\end{bmatrix} + \begin{bmatrix} y_1 \\ y_2\end{bmatrix}) = (x_1 + y_1) + i(x_2 + y_2) = x_1 + ix_2 + y_1 + iy_2 \\ =\Phi(\begin{bmatrix} x_1 \\ x_2\end{bmatrix}) + \Phi(\begin{bmatrix} y_1 \\ y_2\end{bmatrix})
$$

$$
\Phi(\lambda \begin{bmatrix} x_1 \\ x_2\end{bmatrix}) = \lambda x_1 + \lambda i x_2 = \lambda(x_1 + ix_2) = \lambda \Phi(\begin{bmatrix} x_1 \\ x_2\end{bmatrix})
$$



**Theorem** 2.17

*Finite-dimensional vector spaces* V *and* W *are isomorphic if and only if* dim(V ) = dim(W )*.*

위의 theorem을 바탕으로 알 수 있는 것은 다음과 같다.

- 같은 차원의 vector space는 서로 어떤 loss도 없이 서로의 공간으로 mapping할 수 있다.

- $R^{n \times m}, R^{nm}$은 둘 간의 linear, bijective mapping 존재한다. 



## Matrix representation of Linear mapping

n차원의 vector는 n차원 vector에 대해서 isomorphic 합니다.



coordinate는 다음과 같이 정의됩니다.

vector space V와 ordered basis B가 주어졌을 때, 어떤 $x \in V$든 B의 unique linear combination으로 나타낼 수 있습니다.

- $B = (b_1, b_2, \cdots, b_n)$
- $\alpha_i \in R$

$$
x = \alpha_1 b_1 + \alpha_2 b_2 + \cdots + \alpha_n b_n
$$

이때 coordinate는 아래와 같습니다.
$$
\alpha = \begin{vmatrix} \alpha_1 \\ \vdots \\ \alpha_n\end{vmatrix}
$$


- Reference: https://www.youtube.com/watch?v=P2LTAUO1TdA&ab_channel=3Blue1Brown

## Matrix에 대한 두 가지 관점

- collection of vectors

- Linear mapping

  



## Transformation Matrix

vector space V, W가 있을 때, 각각의 ordered basis를 아래와 같이 정의할 수 있다.

- $B=(b_1, b_2, \cdots, b_n)$
- $C=(c_1, c_2, \cdots, c_m)$

그리고 다음과 같은 linear mapping $\Phi$이 있다고 해보자.

- $\alpha_i \in R$

$$
\Phi: V \rightarrow W
$$

$$
\Phi(b_j) = \alpha_1 c_1 + \alpha_2 c_2 + \cdots + \alpha_m c_m
$$

위의 함수를 해석해보면, V의 basis vector는 W의 basis vector의 linear combination으로 표현된다는 것이다.

