---
title: "Image and Kernel 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Image and Kernel

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/KerIm_2015Joz_L2.png/346px-KerIm_2015Joz_L2.png)

vector space V에서 vector space W로 가는 함수가 있다고 할 때, image는 V가 W에서 mapping 될 수 있는 space를 의미한다. 또한 kernel은 W에서 0 vector에 mapping 되는 V에서의 vector space이다.



### Definition

- $V$: vector space, domain
- $W$: vector space, codomain

$$
ker(\Phi) := \Phi^{-1}(0_W) = \{ v \in V: \Phi(v)^{-1} = 0_W \}
$$

$$
Im(\Phi):=\Phi(V) = \{w \in W| v \in V: \Phi(v) = w\}
$$



## Properties

- 항상 $\Phi(0_V) = 0_W$이다. 따라서 kernel은 항상 빈 공간이 아니다.
- $Im(\Phi)$는 W의 vector subspace이고, $ker(\Phi)$는 V의 vector subspace이다.
- 함수 $\Phi$는 $ker = \{0_V\}$인 경우에만 injective function이다. (One-to-One)



## Column space와 null space

- $A \in R^{mxn}$
- $\Phi: x \rightarrow Ax$

$$
Im(\Phi) = \{ Ax : x \in R^m\} = \{ \sum_{i=1}^m \alpha_ix_i: x_i \in R\} = span[a_1, \cdots, a_m]
$$

A의 Column space는 결국 image공간과 일치한다. 따라서 $rk(A) = dim(Im(\Phi))$이다.

그리고, kernel space는 $Ax = 0$의 해의 공간이다.



## Rank nullity thoerem

- $\Phi: V \rightarrow W$

$$
dim(ker(\Phi)) + dim(im(\Phi)) = dim(V)
$$



- 만약 $dim(im(\Phi)) < V$보다 작다면, $dim(ker(\Phi)) \ge 1$일것이다.
- 즉, $Ax = 0$의 해가 무수히 많을 것이다.
- 만약 $dim(im(\Phi)) = V$이라면, 아래의 세 가지 성격을 모두 만족한다.
  - $\Phi$: injective, surjective, bijective







