---
title: "Basis change 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Basis Change

![]({{ site.baseurl }}/images/2020-09-09-basis-change/basis_change.png)

linear mapping $\Psi: V \rightarrow W$ 와 ordered basis $B, \hat{B} \in V$,  $C, \hat{C} \in W$가 주어졌을 때,

- $B = (b_1, b_2, \cdots, b_n)$
- $\hat{B} = (\hat{b}_1, \hat{b}_2, \cdots, \hat{b}_n)$
- $C = (c_1, c_2, \cdots, c_n)$
- $\hat{C} = (\hat{c}_1, \hat{c}_2, \cdots, \hat{c}_n)$

transformation matrix $A_{\Psi}: B \rightarrow C$가 있다. 이 때, $\hat{A}_{\Phi}: \hat{B} \rightarrow \hat{C}$는 아래와 같이 나타낼 수 있다.

- $T: \hat{B} \rightarrow B$
- $S: \hat{C} \rightarrow C$

$$
\hat{A}_{\Phi} = T^{-1}A_{\Phi}S
$$



### Equivalence

$A, \hat{A}$는 다음과 같은 조건을 만족하면, equivalent하다.

아래의 수식이 성립

- $S \in R^{nxn}, T \in R^{mxm}$

$$
\hat{A} = T^{-1}AS
$$

### Similarity

$A, \hat{A}$는 다음과 같은 조건을 만족하면, similar하다.

아래의 수식이 성립

- $S \in R^{nxn}$

$$
\hat{A} = S^{-1}AS
$$





