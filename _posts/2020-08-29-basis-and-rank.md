---
title: "basis and rank 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---

## Generating set and span

vector space V가 있다고 가정할 때, vector의 집합 $A = \{ x_1, \cdots, x_k} \subset V$를 정의할 수 있다. 만약 A의 linear combination으로 V를 모두 표현할 수 있다면, A를 V의 generating set이라고 표현하며, 모든 A의 linear combination의 집합은 A의 span이라고 한다.

## Basis

vector space V가 있다고 가정할 때, vector의 집합 $A = \{ x_1, \cdots, x_k} \subset V$를 정의할 수 있다. 만약 다음과 같은 조건을 충족할 수 있다면, A는 V의 basis이다.

- A보다 더 작으면서, V의 generating set이 될 수 있는 집합이 없다.

따라서, 다음의  네가지 명제는 모두 같은 것을 의미한다.

- A가 V의 basis이다.
- A는 V의 minimal generating set이다.
- A는 V의 maximal linear independent vector이다.
- V의 vector들은 모두 A의 linear combination으로 표현할 수 있다.



## 참고자료

- [Linear combinations, span, and basis vectors](https://www.youtube.com/watch?v=k7RM-ot2NWY&ab_channel=3Blue1Brown)



## Rank

matrix $A \in R^{m^n}$의 linear independent한 column의 개수가 rank(A)이다. 그리고 linear independent한 column의 개수와 row의 개수는 같다.

### Proof: using orthogonality

$A \in R^{mn}$이 있다고 가정해보자. 그리고 row rank가 r이라면, row space의 basis는 다음과 같다.
$$
\{x_1, x_2, \cdots, x_r\} 
$$
그리고 아래의 column vector는 모두 linear independent하다.
$$
Ax_1, Ax_2, \cdots, Ax_r
$$
따라서, column space의 차원은 최소한 row space의 차원만큼을 가지는 것이 확인되었다.

그리고 이 과정을 $A^T$에도 한다면, row space의 차원이 column spcae의 차원보다 크거다 같다는 결론이 나오므로, column space의 rank와 row space의 rank는 같다.

- https://en.wikipedia.org/wiki/Rank_(linear_algebra)#Proofs_that_column_rank_=_row_rank

Rank는 아래와 같은 성질을 가진다.

- $rank(A) = rank(A^T)$
- $A \in R^{mn}$의 column의 subspace의 차원과 rank(A)는 같으며, subspace는 image또는 range라고 부른다.
- $A \in R^{mn}$의 row의 subspace의 차원과 rank(A)는 같다.
- A는 full rank일때만, 역행렬을 가진다. (rk(A) = min(m, n))
- linear equation system $Ax = B$가 해를 가질려면, $rank(A) = rank(A \mid B)$이어야 한다. ('|'은  augmented system을 의미)
- linear equation system $Ax=0$의 해는 n - rank(A)만큼의 null space를 가진다.
- full rank: rk(A) = min(m, n)





