---
title: "2021-01-22-Construction-of-a-Probability-Space.md"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---



데이터에 대한 불확실성의 확률분포를 정의한다면, 고려할 법칙은 Sum Rule과 Product Rule뿐이다.



아래의 수식은 Joint Probability를 나타낸다. $x$와 $y$가 동시에 일어나는 확률분포이다.
$$
p(x, y)
$$
아래의 수식들은 Marginal Probability이다.
$$
p(x), p(y)
$$




아래의 수식은 Conditional Probability를 나타낸다. $x$ 가 주어졌을 때,  $y$의 확률분포이다.
$$
p(y \mid x)
$$


Marginal Distribution과 Conditional Distribution을 활용하여, 위의 두가지 Rule에 대해서 설명할 것이다.



## Sum Rule

- $\mathcal{Y}$: Target Space of Random Variable $Y$

$$
p(x) =
\begin{cases}
\sum_{y \in \mathcal{Y}} p(x, y), & \mbox{if }y\mbox{ is discrete} \\
\int_{\mathcal{Y}} p(x,y) dy, & \mbox{if }y\mbox{ is continuous}
\end{cases}
$$

- $x = [x_1, \cdots, x_D] ^T$

$$
p(x_i) = \int p(x_1, \cdots, x_D) dx_{/i}
$$





## Product Rule	

$$
p(x, y) = p(y \mid x) p(x)
$$





## Bayes's Rule

머신러닝에서는 관측한 Random Variable을 바탕으로 관측되지 않은(Latent) Random Variable로 Inference를 해야한다. 사전지식을 $p(x)$라고 해보자. 이 사전지식은 관측되지 않는 Random Variable $x$에 대한 것이다. 그리고 관측가능한 Second Random Variable을 $y$라고 하겠다.

- $p(y \mid x)$: Likelihood
- $p(x)$: Prior
- $p(y)$: Evidence

$$
p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}
$$





Likelihood $p(x \mid y)$는 $x$라는 것이 관측되었을 때 $y$의 확률분포를 의미한다. 따라서 Likelihood는 $y$에 대한 확률분포이 $x$에 대한 것이 아니다.

Prior $p(x)$는 $x$에 대한 확률분포이며, 모든 $x$에 대해서 Pdf, Pmf가 Nonzero 특성을 가져야한다 (매우 작은 값을 가지더라도).

Posterior $p(x \mid y)$는 Bayesian Statistics에서 관심을 가지는 값인데, 해석해보면, $y$를 관측하고나서 $x$에 대한 정보이다. 머신러닝에서는 일반적으로 모델이 $x$의 역할을 한다. 이를 해석해보면, 데이터가 주어졌을 때, 이 모델이 어떤 확률분포를 가지고 있는가에 대한 것이다. 즉, 모델이 얼마나 신뢰할 수 있는지에 대한 정보를 제공할 수 있다.



아래의 수식은 Marginal Likelihood/Evidence이다. 모든 $x$에 대하여 Sum을 하기 때문에, $x$와 독립적이다. 그리고 Posterior를 Normalize하는 역할을 한다. 하지만, 이는 Integration 때문에 실제로 구하기 어렵다는 단점이 있다.
$$
p(y) = \int p(y\mid x) p(x) dx = \mathbb{E}_x(p(y \mid x))
$$


