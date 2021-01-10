---
title: "Gradient of Matrix"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---



## Gradient of Matrix

- Input: matrix 형태
- output: matrix 형태

$$
f: R^{m \times n} -> R^{p \times q}
$$

위와 같은 함수가 주어졌을 때, gradient를 구해보자. 결론적으로 jacobian은 $(m \times n) \times (p \times q)$의 형태를 가진다.
$$
J_{ijkl} = \frac{\partial A_{ij}}{\partial B_{kl}}
$$
또한, matrix가 linear mapping이라는 것을 고려해보면, matrix $R^{m \times n}$와 vector $R^{mn}$ 사이의 vector isomorphism이 있다는 것을 알 수 있다. 즉, matrix를 vector의 형태로 나타낼 수 있다는 것이다.

그러므로, 위의 matrix $R^{m \times n}$을 vector $R^{mn}$의 형태로 나타내고, 이를 바탕으로 jacobian을 구하게 되면, $mn \times pq$의 size를 가지게 된다.

