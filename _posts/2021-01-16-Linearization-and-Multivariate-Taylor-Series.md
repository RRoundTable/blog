---
title: "Linearization and Multivariate Taylor Series"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post

---





## Linearization and Multivariate Taylor Series

gradient $\nabla f$는 함수$x_0$에서의 $f$ 를 Locally Approximation 하기 위해서 사용되기도 한다.
$$
f(x) \approx f(x_0)+(\nabla_x f)(x_0)(x - x_0)
$$



아래의 이미지를 보면, 하나의 예시를 알 수 있다. 함수 $f$에 대해서 $x_0$에서의 근사는 아래의 이미지처럼 직선의 형태로 나타낼 수 있다.

![]({{ site.baseurl }}/images/2021-01-16-Linearization-and-Multivariate-Taylor-Series/1.png)



위의 근사는 $x_0$지점에서 정확할 수 있으나, 조금만 벗어나도 격차가 나기 시작합니다. 앞으로 다룰 내용은 조금 더 일반적인 근사에 대해서 다루겠다.



### Definition 5.7: Multivariate Taylor Series

아래와 같은 함수 $f$가 있다고 가정하겠다. 그리고 $f$는 $x_0$에서 Smooth하다.(미분 가능하다)
$$
f: R^D \rightarrow R \\
x \mapsto f(x) , x \in R^D
$$



그리고 Difference Vector를 아래와 같이 정의해보겠다.
$$
\delta = x - x_0
$$



이제 Multivariate Taylor Series는 아래와 같이 정의할 수 있다.

- $D^k_x f(x_0)$: k-th total derivative of f with respect to $x$, evaluated $x_0$

$$
f(x) = \sum_{k=0}^{\infty} \frac{D^k_x f(x_0)}{k!} \delta^k
$$





### Definition 5.8: Taylor Polynomial

- $D^k_x f(x_0)$: k-th total derivative of f with respect to $x$, evaluated $x_0$

$$
T_n(x) = \sum_{k=0}^{n} \frac{D^k_x f(x_0)}{k!} \delta^k
$$



$D^k_x$와 $\delta^k$는 모두 Higher Order Tensor이다.

아래의 $\delta^k$는 다음과 같은 차원수를 가진다.
$$
\delta^k \in R^{D^k}
$$


하나씩 살펴보겠다. 우선 $\delta^2$를 알아보겠다.
$$
\delta^2 = \delta \otimes \delta, \delta^2[i, j] = \delta[i] \delta[j]
$$



$\delta^3$을 구해보면 다음과 같다.
$$
\delta^3 = \delta \otimes \delta \otimes \delta, \delta^3[i, j, k] = \delta[i] \delta[j] \delta[k]
$$


아래의 이미지는 그 과정을 시각화한 것이다.

![]({{ site.baseurl }}/images/2021-01-16-Linearization-and-Multivariate-Taylor-Series/2.png)





이제 아래의 수식을 살펴보자. 
$$
D^0_x f(x_0) \delta^k = \sum_{i_1=1}^D \cdots \sum_{i_k=1}^D D_x^k f(x_0)[i_1, \cdots, i_k] \delta[i_1] \cdots \delta[i_k]
$$



- $H(x_0)$: Hessian of f, evaluated at $x_0$



만약 $k=0$이라면,
$$
D^k_x f(x_0) \delta^0 = f(x_0) \in R
$$


만약 $k=1$이라면,
$$
D^1_x f(x_0) \delta^1 =  \nabla_x f(x_0)[i] \delta[i]
$$


만약 $k=2$라면,
$$
D^2_x f(x_0) \delta^2 =  \delta^T H(x_0) \delta
$$


만약 $k=3$라면,