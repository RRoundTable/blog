---
title: "Big-O notation 정리하기"
toc: true
badges: true
branch: master
categories: ['algorithm']
---

# Big-O notation 정리하기

> In [computer science](https://en.wikipedia.org/wiki/Computer_science), big O notation is used to [classify algorithms](https://en.wikipedia.org/wiki/Computational_complexity_theory) according to how their running time or space requirements grow as the input size grows.[[3\]](https://en.wikipedia.org/wiki/Big_O_notation#cite_note-quantumcomplexity-3) In [analytic number theory](https://en.wikipedia.org/wiki/Analytic_number_theory), big O notation is often used to express a bound on the difference between an [arithmetical function](https://en.wikipedia.org/wiki/Arithmetic_function) and a better understood approximation; a famous example of such a difference is the remainder term in the [prime number theorem](https://en.wikipedia.org/wiki/Prime_number_theorem). 
>
> reference: https://en.wikipedia.org/wiki/Big_O_notation

# 

## Formal definition

$F$ 는 real or complex valued function이며 , $g$는 real valued function이다. 두 함수는 무한한 양의 실수의 subset으로 정의된다. 따라서 $g(x)$는 x가 충분히 큰 값일때 항상 양수이다.

$$
F(x) = O(g(x)) \ \text{as} \ x \rightarrow \infty
$$

충분히 큰 x값을 가질 수 있을때만, $F(x)$의 절대값의 최대값이 $g(x)$에 양의 상수를 곱한 것을 넘지 못할 때, $F(x) = O(g(x))$라고 표현한다. (Upper Bound)이를 아래 식으로 표현한다.

$$
\rvert F(x) \rvert \le Mg(x) \text{for all } x \ge x_0
$$

이를 간단하게 아래의 이미지로 확인할 수 있다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Big-O-notation.png/300px-Big-O-notation.png">

파란색 선은 $Mg(x)$ 를 빨간색 선은 $F(x)$를 의미한다. $x_0$보다 큰 x값을 가지면, $F(x)$는 항상 $Mg(x)$보다 작으며, 이는 $F(x)$의 upper bound가 $Mg(x)$라는 것을 의미한다.